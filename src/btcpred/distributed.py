"""
distributed.py -- "distributed training within a single GPU" (and gracefully
everywhere else).

What this actually means
------------------------
A single BTC price predictor is far too small to saturate a modern GPU on its
own: each step is a 1800-step sequence over a few hundred channels, so the
device spends most of its life waiting on kernel launches and on the CPU-side
Market Analyser.  The fix is to run **several independent ranks on the same
physical device**.  Each rank:

* owns a different shard of the market tape (see ``MarketSimulator.shard``),
* keeps its own GRU carry state and its own stability filter,
* runs its own CUDA stream, so one rank's kernels overlap another's analyser
  stall,
* and synchronises gradients with the others through NCCL all-reduce (DDP).

The result is one model, trained on N market shards at once, on one GPU.

The same entry point degrades cleanly:

===================  ==============================================
hardware             what you get
===================  ==============================================
1 GPU                N ranks sharing cuda:0 (NCCL), AMP on
k GPUs               ranks spread round-robin over all devices
CPU only             N ranks over gloo, model auto-shrunk
1 CPU / no torch     single process, NumPy backend
===================  ==============================================

Rendezvous uses a file store under the run directory, so no free TCP port and
no internet access are required -- which is exactly what an air-gapped
Snowflake container needs.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    _HAS_TORCH = True
except Exception:                                   # pragma: no cover
    torch = None                                    # type: ignore
    dist = None                                     # type: ignore
    mp = None                                       # type: ignore
    _HAS_TORCH = False


@dataclass
class DistContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cpu"
    backend: str = "none"
    is_distributed: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.is_distributed and dist is not None and dist.is_initialized():
            dist.barrier()

    def all_reduce_mean(self, value: float) -> float:
        if not (self.is_distributed and dist is not None and dist.is_initialized()):
            return float(value)
        t = torch.tensor([float(value)], device=self.device if self.device != "cpu" else "cpu")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item() / self.world_size)

    def cleanup(self) -> None:
        if self.is_distributed and dist is not None and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


# --------------------------------------------------------------------------
def pick_device(local_rank: int) -> str:
    """Round-robin ranks over the visible GPUs; all share one when there is one."""
    if _HAS_TORCH and torch.cuda.is_available():
        n = torch.cuda.device_count()
        return f"cuda:{local_rank % max(1, n)}"
    return "cpu"


def init_process_group(rank: int, world_size: int, rendezvous: str,
                       timeout_s: int = 1800) -> DistContext:
    """Join (or create) the process group; returns a populated context."""
    if world_size <= 1 or not _HAS_TORCH:
        return DistContext(rank=0, world_size=1, local_rank=0,
                           device=pick_device(0),
                           backend="none", is_distributed=False)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = pick_device(rank)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        # Several ranks share one card: cap each so one cannot starve the rest.
        try:
            torch.cuda.set_per_process_memory_fraction(
                max(0.1, min(0.95, 1.0 / world_size * 0.95)),
                device=torch.device(device).index or 0)
        except Exception:
            pass

    import datetime as _dt
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{rendezvous}",
        rank=rank, world_size=world_size,
        timeout=_dt.timedelta(seconds=timeout_s),
    )
    return DistContext(rank=rank, world_size=world_size, local_rank=rank,
                       device=device, backend=backend, is_distributed=True)


# --------------------------------------------------------------------------
def _worker(rank: int, world_size: int, rendezvous: str,
            fn: Callable[..., Any], kwargs: Dict[str, Any]) -> None:
    ctx = None
    try:
        ctx = init_process_group(rank, world_size, rendezvous)
        if ctx.device.startswith("cuda"):
            # Give every rank its own stream so their kernels interleave on the
            # shared device instead of serialising on the default stream.
            with torch.cuda.stream(torch.cuda.Stream(device=ctx.device)):
                fn(ctx, **kwargs)
        else:
            fn(ctx, **kwargs)
    except Exception:                                 # pragma: no cover
        sys.stderr.write(f"[rank {rank}] crashed:\n{traceback.format_exc()}")
        raise
    finally:
        if ctx is not None:
            ctx.cleanup()


def launch(fn: Callable[..., Any], world_size: int = 1,
           rendezvous_dir: Optional[str] = None, **kwargs) -> Any:
    """
    Run ``fn(ctx, **kwargs)`` on ``world_size`` ranks.

    ``world_size == 1`` runs in-process (no spawn, no process group) so the
    single-CPU and notebook paths stay debuggable and fast to start.
    """
    world_size = max(1, int(world_size))
    if world_size == 1 or not _HAS_TORCH:
        ctx = DistContext(rank=0, world_size=1, local_rank=0,
                          device=pick_device(0), backend="none",
                          is_distributed=False)
        return fn(ctx, **kwargs)

    d = rendezvous_dir or tempfile.mkdtemp(prefix="btcpred_rdzv_")
    os.makedirs(d, exist_ok=True)
    rdzv = os.path.join(d, "rendezvous")
    if os.path.exists(rdzv):
        os.remove(rdzv)

    # 'spawn' is required for CUDA; it is also the only safe start method
    # inside a notebook kernel.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_worker, args=(world_size, rdzv, fn, kwargs),
             nprocs=world_size, join=True)
    return None


# --------------------------------------------------------------------------
def describe_plan(world_size: int) -> str:
    if not _HAS_TORCH:
        return "torch absent -> single-process NumPy backend"
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = {torch.cuda.get_device_name(i) for i in range(n)}
        if n == 1:
            return (f"{world_size} ranks sharing 1x {names.pop()} "
                    f"(intra-GPU DDP over NCCL, per-rank CUDA streams, "
                    f"memory fraction {1/world_size:.0%} each)")
        return f"{world_size} ranks over {n} GPUs ({', '.join(names)}) via NCCL"
    return f"{world_size} CPU ranks over gloo"
