"""
env.py -- hardware discovery and automatic model sizing.

The same code has to run on a laptop CPU, a 2 GB Snowflake warehouse kernel and
a multi-GPU container pool.  Rather than making the user guess hyper-parameters,
we measure what is actually available and *derive* the model size, the sequence
length, the batch size and the number of distributed workers from it.

Sizing rule of thumb used below
-------------------------------
Training memory per parameter with Adam ~= 16 bytes
    4 B  fp32 weight
  + 4 B  gradient
  + 8 B  Adam m and v
Activations are bounded separately by ``batch x seq_len x d_model``.
We spend at most ``MEM_FRACTION`` of the smallest relevant memory pool and then
snap down to the nearest predefined tier so runs stay reproducible.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import List, Optional

MEM_FRACTION = 0.55          # share of free memory we are willing to use
BYTES_PER_PARAM_TRAIN = 16   # fp32 weight + grad + Adam moments


# --------------------------------------------------------------------------
@dataclass
class HardwareProfile:
    cpu_count: int
    ram_gb: float
    has_cuda: bool
    n_gpus: int
    gpu_names: List[str] = field(default_factory=list)
    vram_gb: float = 0.0           # per-device VRAM of the smallest device
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    platform: str = ""
    in_snowflake: bool = False
    supports_amp: bool = False
    supports_bf16: bool = False

    @property
    def device(self) -> str:
        return "cuda" if self.has_cuda else "cpu"

    @property
    def compute_budget_gb(self) -> float:
        """Memory pool the model is actually allowed to live in."""
        return (self.vram_gb if self.has_cuda else self.ram_gb) * MEM_FRACTION

    def summary(self) -> str:
        g = (f"{self.n_gpus}x {self.gpu_names[0] if self.gpu_names else 'GPU'} "
             f"({self.vram_gb:.1f} GB VRAM)" if self.has_cuda else "no CUDA")
        return (f"{self.platform} | {self.cpu_count} vCPU | {self.ram_gb:.1f} GB RAM "
                f"| {g} | torch={self.torch_version or 'absent'}"
                + (" | Snowflake" if self.in_snowflake else ""))


@dataclass
class AutoConfig:
    """Everything that scales with the machine."""
    tier: str
    device: str
    d_model: int
    n_layers: int
    n_heads: int
    tcn_channels: int
    seq_len: int             # seconds of history fed to the predictor
    batch_streams: int       # parallel market streams per worker
    world_size: int          # distributed workers (processes)
    grad_accum: int
    amp_dtype: Optional[str]
    analyser_workers: int
    est_params: int
    est_train_mem_gb: float
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Tiers are ordered small -> large.  (d_model, layers, heads, tcn, seq_len)
TIERS = [
    ("pico",  48,  2, 2,  32,  128),
    ("nano",  64,  2, 4,  48,  256),
    ("micro", 96,  3, 4,  64,  384),
    ("small", 128, 3, 4,  96,  512),
    ("base",  192, 4, 6, 128,  768),
    ("large", 256, 6, 8, 192, 1024),
    ("xl",    384, 8, 8, 256, 1536),
    ("xxl",   512, 10, 16, 384, 1800),
]


# --------------------------------------------------------------------------
def _ram_gb() -> float:
    # cgroup v2 (containers / Snowflake) first -- /proc/meminfo lies in a cgroup
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(p) as fh:
                v = fh.read().strip()
            if v not in ("max", "") and int(v) < (1 << 50):
                return int(v) / 1e9
        except Exception:
            pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 4.0


def _cpu_count() -> int:
    try:
        q = os.cpu_count() or 2
        # respect cgroup cpu quota if present
        try:
            with open("/sys/fs/cgroup/cpu.max") as fh:
                quota, period = fh.read().split()
            if quota != "max":
                q = max(1, min(q, int(int(quota) / int(period))))
        except Exception:
            pass
        return max(1, q)
    except Exception:
        return 2


def _nvidia_smi_vram() -> List[float]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return [float(x.strip()) / 1024.0 for x in out.split("\n") if x.strip()]
    except Exception:
        return []


def detect() -> HardwareProfile:
    prof = HardwareProfile(
        cpu_count=_cpu_count(),
        ram_gb=_ram_gb(),
        has_cuda=False,
        n_gpus=0,
        platform=f"{platform.system()} {platform.machine()} py{platform.python_version()}",
        in_snowflake=any(k in os.environ for k in
                         ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_HOST",
                          "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_SERVICE_NAME")),
    )
    try:
        import torch
        prof.torch_version = torch.__version__
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            prof.has_cuda = True
            prof.n_gpus = torch.cuda.device_count()
            names, vrams = [], []
            for i in range(prof.n_gpus):
                p = torch.cuda.get_device_properties(i)
                names.append(p.name)
                vrams.append(p.total_memory / 1e9)
            prof.gpu_names = names
            prof.vram_gb = min(vrams)
            prof.cuda_version = torch.version.cuda
            prof.supports_amp = True
            try:
                prof.supports_bf16 = torch.cuda.is_bf16_supported()
            except Exception:
                prof.supports_bf16 = False
    except ImportError:
        vr = _nvidia_smi_vram()
        if vr:
            prof.n_gpus = len(vr)
            prof.vram_gb = min(vr)
    return prof


# --------------------------------------------------------------------------
def _params_for(d_model: int, n_layers: int, tcn: int, n_feat: int) -> int:
    """Rough parameter count of the PricePredictor at a given size."""
    tcn_p = n_feat * tcn * 5 + tcn * tcn * 5 * 2          # 3 dilated conv blocks
    gru_p = 3 * (d_model * d_model + d_model * tcn + 2 * d_model)
    attn_p = n_layers * (4 * d_model * d_model + 8 * d_model * d_model)
    head_p = d_model * 128 + 128 * 3
    return int(tcn_p + gru_p + attn_p + head_p)


def autoscale(prof: Optional[HardwareProfile] = None,
              n_features: int = 40,
              max_tier: Optional[str] = None,
              min_tier: str = "pico",
              force_world_size: Optional[int] = None) -> AutoConfig:
    """Pick the largest tier that comfortably fits the detected hardware."""
    prof = prof or detect()
    budget_gb = max(0.15, prof.compute_budget_gb)

    names = [t[0] for t in TIERS]
    lo = names.index(min_tier) if min_tier in names else 0
    hi = names.index(max_tier) if max_tier in names else len(TIERS) - 1

    # Memory is not the only ceiling: on CPU the binding constraint is FLOPs.
    # A 15 M-parameter sequence model "fits" in 2 GB of RAM and still needs
    # minutes per window on two cores, so cap the tier by core count too.
    if not prof.has_cuda:
        cpu_idx = int(math.floor(math.log2(max(1, prof.cpu_count))))
        hi = min(hi, max(0, cpu_idx))
    else:
        # ~1 tier per doubling of VRAM beyond 4 GB, floor at 'small'
        vram_idx = 3 + int(math.floor(math.log2(max(1.0, prof.vram_gb / 4.0))))
        hi = min(hi, max(0, min(len(TIERS) - 1, vram_idx)))

    chosen = TIERS[lo]
    for t in TIERS[lo:hi + 1]:
        name, d, L, H, tcn, seq = t
        p = _params_for(d, L, tcn, n_features)
        mem = p * BYTES_PER_PARAM_TRAIN / 1e9
        if mem < budget_gb * 0.35:          # leave 65 % for activations + data
            chosen = t
        else:
            break

    name, d, L, H, tcn, seq = chosen
    params = _params_for(d, L, tcn, n_features)

    # ---- distributed workers --------------------------------------------
    if force_world_size:
        world = max(1, int(force_world_size))
    elif prof.has_cuda:
        # "distributed training within a single GPU": several ranks share one
        # device so copy/compute overlap and the SMs stay saturated.
        per_rank_gb = max(0.75, params * BYTES_PER_PARAM_TRAIN / 1e9 * 3)
        world = int(max(1, min(8, prof.vram_gb * MEM_FRACTION // per_rank_gb)))
        world = max(1, min(world, max(1, prof.cpu_count)))
    else:
        world = max(1, min(4, prof.cpu_count // 2))

    # ---- activation-bounded batch ---------------------------------------
    act_budget = budget_gb * 0.45 * 1e9 / max(1, world)
    bytes_per_stream = seq * d * 4 * 12          # 12 cached tensors/step, fp32
    batch = int(max(1, min(64, act_budget // max(1, bytes_per_stream))))
    if not prof.has_cuda:
        # lanes are pure extra compute on a CPU; keep them proportional to cores
        batch = int(max(1, min(batch, max(1, prof.cpu_count))))

    amp = None
    if prof.has_cuda:
        amp = "bf16" if prof.supports_bf16 else "fp16"

    analyser_workers = max(1, min(prof.cpu_count - (1 if prof.has_cuda else 0),
                                  max(1, prof.cpu_count // 2)))

    est_mem = (params * BYTES_PER_PARAM_TRAIN + batch * bytes_per_stream) / 1e9 * world
    note = ("GPU detected: {w} ranks share cuda:0 (intra-GPU DDP), amp={a}"
            if prof.has_cuda else
            "No GPU: {w} CPU ranks via gloo; model auto-shrunk to '{t}'")
    return AutoConfig(
        tier=name, device=prof.device, d_model=d, n_layers=L, n_heads=H,
        tcn_channels=tcn, seq_len=seq, batch_streams=batch, world_size=world,
        grad_accum=1, amp_dtype=amp, analyser_workers=analyser_workers,
        est_params=params, est_train_mem_gb=round(est_mem, 3),
        notes=note.format(w=world, a=amp, t=name),
    )


def describe(prof: Optional[HardwareProfile] = None,
             cfg: Optional[AutoConfig] = None) -> str:
    prof = prof or detect()
    cfg = cfg or autoscale(prof)
    return (f"[hardware] {prof.summary()}\n"
            f"[autoscale] tier={cfg.tier} d_model={cfg.d_model} layers={cfg.n_layers} "
            f"seq_len={cfg.seq_len}s streams={cfg.batch_streams} "
            f"world_size={cfg.world_size} amp={cfg.amp_dtype}\n"
            f"[autoscale] ~{cfg.est_params/1e6:.2f} M params, "
            f"~{cfg.est_train_mem_gb:.2f} GB training footprint -- {cfg.notes}")


if __name__ == "__main__":
    p = detect()
    c = autoscale(p)
    print(describe(p, c))
    print(json.dumps({"hardware": asdict(p), "autoconfig": c.to_dict()}, indent=2))
