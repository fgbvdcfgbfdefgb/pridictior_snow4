"""
model.py -- the **Price Predictor** (GPU component).

Given the Market Analyser's signal vector for second *t*, predict the BTC price
**1800 seconds (30 minutes) ahead**.

Design notes
------------
*   The network predicts a **log return** ``m = 1000 * ln(P[t+1800] / P[t])``
    rather than a price.  That is scale-free: a model trained on $6k Bitcoin in
    2017 remains valid at $120k in 2026, and the loss is not dominated by the
    price level.
*   It is **strictly causal** (left-padded dilated convolutions + a GRU), so
    running a whole 1800-second window in one batched forward pass gives
    exactly the same numbers as stepping one second at a time -- that identity
    is what makes "max speed" training legitimate.
*   The GRU hidden state is **carried across windows**, so the model has
    continuous memory of the market rather than restarting every epoch.
*   Three heads: ``mu`` (the forecast), ``log_sigma`` (predicted uncertainty,
    trained with a Gaussian NLL so the model can say "I don't know"), and a
    direction logit (an auxiliary task that sharpens the sign of the move).
*   Two backends behind one interface:
      - ``TorchPredictor``  -- CUDA / AMP / DDP, used whenever torch exists;
      - ``NumpyPredictor``  -- a hand-differentiated MLP that needs nothing but
        NumPy, so the pipeline still trains on a bare Snowflake warehouse
        kernel with no deep-learning stack installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:                                    # torch is optional on purpose
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                       # pragma: no cover
    torch = None                        # type: ignore
    nn = object                         # type: ignore
    _HAS_TORCH = False


HORIZON = 1800              # seconds ahead
RET_SCALE = 1000.0          # work in per-mille log-return units


# ==========================================================================
# loss configuration
# ==========================================================================
@dataclass
class LossConfig:
    huber_delta: float = 2.0        # per-mille; ~0.2 % move
    nll_weight: float = 0.25        # Gaussian NLL on (mu, sigma)
    direction_weight: float = 0.15  # auxiliary sign classification
    smooth_weight: float = 0.60     # penalise second-to-second jitter
    smooth2_weight: float = 0.20    # penalise curvature (2nd difference)
    reward_weight: float = 0.35     # emphasise seconds with large real moves
    move_floor: float = 0.25        # per-mille, below this a move is noise


# ==========================================================================
# Torch implementation
# ==========================================================================
if _HAS_TORCH:

    class CausalConvBlock(nn.Module):
        """Left-padded dilated conv + GELU + residual.  Zero future leakage."""

        def __init__(self, c_in: int, c_out: int, k: int, dilation: int):
            super().__init__()
            self.pad = (k - 1) * dilation
            self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)
            self.norm = nn.GroupNorm(1, c_out)
            self.act = nn.GELU()
            self.res = (nn.Identity() if c_in == c_out
                        else nn.Conv1d(c_in, c_out, 1))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":   # (B, C, T)
            y = F.pad(x, (self.pad, 0))
            y = self.act(self.norm(self.conv(y)))
            return y + self.res(x)

    class PricePredictorNet(nn.Module):
        """Causal TCN -> GRU -> (mu, log_sigma, direction)."""

        def __init__(self, n_features: int, d_model: int = 128,
                     n_layers: int = 3, tcn_channels: int = 96,
                     dropout: float = 0.05):
            super().__init__()
            self.n_features = n_features
            self.d_model = d_model
            self.inp = nn.Linear(n_features, tcn_channels)
            self.tcn = nn.ModuleList([
                CausalConvBlock(tcn_channels, tcn_channels, 5, 2 ** i)
                for i in range(max(1, n_layers))
            ])
            self.gru = nn.GRU(tcn_channels, d_model, num_layers=1,
                              batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, 3),
            )
            # start life predicting "no change with wide error bars"
            with torch.no_grad():
                self.head[-1].weight.mul_(0.01)
                self.head[-1].bias.zero_()
                self.head[-1].bias[1] = 1.0          # log_sigma ~ e^1 per-mille

        def forward(self, x: "torch.Tensor",
                    h: Optional["torch.Tensor"] = None):
            """x: (B, T, F) -> mu (B,T), log_sigma (B,T), dir (B,T), h"""
            z = self.inp(x).transpose(1, 2)              # (B, C, T)
            for blk in self.tcn:
                z = blk(z)
            z = z.transpose(1, 2)                        # (B, T, C)
            z, h = self.gru(z, h)
            o = self.head(self.drop(z))
            mu = o[..., 0]
            log_sigma = o[..., 1].clamp(-4.0, 4.0)
            direction = o[..., 2]
            return mu, log_sigma, direction, h


# ==========================================================================
# common interface
# ==========================================================================
class BasePredictor:
    backend = "none"

    def n_params(self) -> int: ...
    def predict(self, feats: np.ndarray, carry: Any = None): ...
    def learn(self, feats: np.ndarray, target: np.ndarray, carry: Any = None,
              weights: Optional[np.ndarray] = None) -> Dict[str, float]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, d: dict) -> None: ...


class TorchPredictor(BasePredictor):
    """Torch/CUDA backend, optionally wrapped in DDP by ``distributed.py``."""

    backend = "torch"

    def __init__(self, n_features: int, cfg, device: str = "cpu",
                 loss_cfg: Optional[LossConfig] = None, lr: float = 2e-3,
                 ddp: bool = False, amp_dtype: Optional[str] = None,
                 compile_model: bool = False):
        if not _HAS_TORCH:                               # pragma: no cover
            raise RuntimeError("torch is not available")
        self.device = torch.device(device)
        self.loss_cfg = loss_cfg or LossConfig()
        self.net = PricePredictorNet(
            n_features, d_model=cfg.d_model, n_layers=cfg.n_layers,
            tcn_channels=cfg.tcn_channels).to(self.device)
        self.raw_net = self.net
        if compile_model and hasattr(torch, "compile"):
            try:
                self.net = torch.compile(self.net)       # pragma: no cover
            except Exception:
                pass
        if ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.net = DDP(self.net,
                           device_ids=[self.device.index] if self.device.type == "cuda" else None,
                           output_device=self.device.index if self.device.type == "cuda" else None,
                           find_unused_parameters=False)
        self.opt = torch.optim.AdamW(self.raw_net.parameters(), lr=lr,
                                     betas=(0.9, 0.98), weight_decay=1e-4,
                                     eps=1e-8)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=64, T_mult=2, eta_min=lr * 0.05)
        self.amp_dtype = None
        if amp_dtype == "bf16":
            self.amp_dtype = torch.bfloat16
        elif amp_dtype == "fp16":
            self.amp_dtype = torch.float16
        self.scaler = (torch.amp.GradScaler("cuda")
                       if self.amp_dtype == torch.float16 else None)
        self.steps = 0

    # -- helpers -----------------------------------------------------------
    def n_params(self) -> int:
        return sum(p.numel() for p in self.raw_net.parameters())

    def _to(self, a: np.ndarray) -> "torch.Tensor":
        return torch.as_tensor(np.ascontiguousarray(a), device=self.device)

    def _autocast(self):
        if self.amp_dtype is not None and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=self.amp_dtype)
        import contextlib
        return contextlib.nullcontext()

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def predict(self, feats: np.ndarray, carry: Any = None):
        """feats (T, F) or (B, T, F) -> mu, sigma (same leading shape), carry."""
        self.raw_net.eval()
        x = self._to(feats.astype(np.float32))
        squeeze = (x.dim() == 2)
        if squeeze:
            x = x.unsqueeze(0)
        h = carry if isinstance(carry, torch.Tensor) else None
        with self._autocast():
            mu, log_sigma, direction, h = self.raw_net(x, h)
        mu = mu.float()
        sigma = log_sigma.float().exp()
        if squeeze:
            mu, sigma = mu[0], sigma[0]
        return (mu.cpu().numpy(), sigma.cpu().numpy(),
                h.detach() if h is not None else None)

    # -- training -----------------------------------------------------------
    def learn(self, feats: np.ndarray, target: np.ndarray, carry: Any = None,
              weights: Optional[np.ndarray] = None,
              grad_clip: float = 1.0) -> Dict[str, float]:
        """
        One optimisation step over a whole 30-minute window.

        ``feats``  (T, F) or (B, T, F) features of the window being *scored*
        ``target`` per-mille log return actually realised 1800 s later
        """
        self.raw_net.train()
        c = self.loss_cfg
        x = self._to(feats.astype(np.float32))
        y = self._to(target.astype(np.float32))
        if x.dim() == 2:
            x, y = x.unsqueeze(0), y.unsqueeze(0)
        w = (self._to(weights.astype(np.float32)) if weights is not None
             else torch.ones_like(y))
        if w.dim() == 1:
            w = w.unsqueeze(0)
        h = carry if isinstance(carry, torch.Tensor) else None

        self.opt.zero_grad(set_to_none=True)
        with self._autocast():
            mu, log_sigma, direction, h_out = self.raw_net(x, h)
            err = mu - y
            # --- accuracy term: weighted Huber ---------------------------
            a = err.abs()
            huber = torch.where(a <= c.huber_delta, 0.5 * err ** 2,
                                c.huber_delta * (a - 0.5 * c.huber_delta))
            # reward weighting: seconds with a real move matter more
            rw = 1.0 + c.reward_weight * (y.abs() / (c.move_floor + y.abs().mean() + 1e-6))
            acc_loss = (huber * w * rw).mean()
            # --- calibrated uncertainty ----------------------------------
            inv = torch.exp(-2.0 * log_sigma)
            nll = 0.5 * (err ** 2) * inv + log_sigma
            nll_loss = (nll * w).mean()
            # --- direction (auxiliary) ------------------------------------
            big = (y.abs() > c.move_floor).float()
            dir_loss = (F.binary_cross_entropy_with_logits(
                direction, (y > 0).float(), reduction="none") * big * w).sum() \
                / (big * w).sum().clamp_min(1.0)
            # --- stability: the forecast must not twitch -------------------
            d1 = mu[:, 1:] - mu[:, :-1]
            smooth = (d1 ** 2).mean()
            d2 = d1[:, 1:] - d1[:, :-1]
            curve = (d2 ** 2).mean()
            loss = (acc_loss + c.nll_weight * nll_loss
                    + c.direction_weight * dir_loss
                    + c.smooth_weight * smooth + c.smooth2_weight * curve)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            gn = torch.nn.utils.clip_grad_norm_(self.raw_net.parameters(), grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.raw_net.parameters(), grad_clip)
            self.opt.step()
        self.sched.step()
        self.steps += 1

        with torch.no_grad():
            mae = err.abs().mean()
            dir_acc = (((mu > 0) == (y > 0)).float() * big).sum() / big.sum().clamp_min(1.0)
        return {
            "loss": float(loss.detach()),
            "acc_loss": float(acc_loss.detach()),
            "nll": float(nll_loss.detach()),
            "dir_loss": float(dir_loss.detach()),
            "smooth": float(smooth.detach()),
            "curve": float(curve.detach()),
            "mae_permille": float(mae),
            "dir_acc": float(dir_acc),
            "grad_norm": float(gn),
            "lr": float(self.opt.param_groups[0]["lr"]),
        }

    # -- persistence ---------------------------------------------------------
    def state_dict(self) -> dict:
        return {"backend": "torch", "steps": self.steps,
                "net": {k: v.detach().cpu() for k, v in
                        self.raw_net.state_dict().items()},
                "opt": self.opt.state_dict()}

    def load_state_dict(self, d: dict) -> None:
        # Checkpoints round-trip through .npz so they stay readable without
        # torch installed, which means tensors come back as NumPy arrays.
        net = {k: (v if hasattr(v, "detach") else torch.as_tensor(np.asarray(v)))
               for k, v in d["net"].items()}
        ref = self.raw_net.state_dict()
        net = {k: v.to(dtype=ref[k].dtype) if k in ref else v
               for k, v in net.items()}
        self.raw_net.load_state_dict(net)
        if "opt" in d and isinstance(d["opt"], dict) and "state" in d["opt"]:
            try:
                self.opt.load_state_dict(d["opt"])
            except Exception:                            # pragma: no cover
                pass
        self.steps = int(np.asarray(d.get("steps", 0)).item()
                         if not isinstance(d.get("steps", 0), int)
                         else d.get("steps", 0))


# ==========================================================================
# Pure-NumPy fallback -- no torch, no BLAS tricks, still really trains
# ==========================================================================
class NumpyPredictor(BasePredictor):
    """
    Hand-differentiated 2-layer MLP over the analyser features plus three
    EMA-pooled "context" copies of them (a cheap stand-in for recurrence).

    Trained with Adam on the same objective as the torch model (Huber +
    smoothness).  It is small, but it is a genuine learner, and it means the
    repository trains end-to-end on a stock Snowflake warehouse kernel where
    only NumPy is guaranteed.
    """

    backend = "numpy"
    CTX_SPANS = (15, 120, 900)

    def __init__(self, n_features: int, cfg, loss_cfg: Optional[LossConfig] = None,
                 lr: float = 3e-3, seed: int = 0):
        self.loss_cfg = loss_cfg or LossConfig()
        self.nf = n_features
        self.d_in = n_features * (1 + len(self.CTX_SPANS)) + 1
        self.d_h = max(32, min(256, cfg.d_model))
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((self.d_in, self.d_h))
                   / math.sqrt(self.d_in)).astype(np.float64)
        self.b1 = np.zeros(self.d_h)
        self.W2 = (rng.standard_normal((self.d_h, 2)) * 0.01).astype(np.float64)
        self.b2 = np.array([0.0, 1.0])
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.lr = lr
        self.steps = 0
        self._ctx = {s: np.zeros(n_features) for s in self.CTX_SPANS}

    def _params(self) -> Dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def n_params(self) -> int:
        return int(sum(v.size for v in self._params().values()))

    # -- context expansion (stateful, causal) --------------------------------
    def _expand_batch(self, feats: np.ndarray, carry: Any, advance: bool):
        """
        Accept (F,), (T, F) or (B, T, F).

        For a batch, each element is expanded independently from the same
        incoming context and then concatenated, with ``seam`` marking the first
        row of each element so the smoothness penalty never spans two
        unrelated stretches of tape.
        """
        f = np.asarray(feats, dtype=np.float64)
        if f.ndim == 3:
            Xs, seams, ctx_out = [], [], None
            for b in range(f.shape[0]):
                X, c = self._expand(f[b], carry, advance=advance and b == 0)
                Xs.append(X)
                s = np.zeros(X.shape[0], dtype=bool)
                s[0] = True
                seams.append(s)
                ctx_out = c if b == 0 else ctx_out
            return np.concatenate(Xs, 0), np.concatenate(seams), ctx_out
        X, c = self._expand(f, carry, advance)
        seam = np.zeros(X.shape[0], dtype=bool)
        seam[0] = True
        return X, seam, c

    def _expand(self, feats: np.ndarray, carry: Any, advance: bool):
        from .analyser import ema_run
        f = np.asarray(feats, dtype=np.float64)
        if f.ndim == 1:
            f = f[None, :]
        T = f.shape[0]
        ctx = dict(carry) if isinstance(carry, dict) else {
            s: np.zeros(self.nf) for s in self.CTX_SPANS}
        parts = [f]
        new_ctx = {}
        for s in self.CTX_SPANS:
            a = 2.0 / (s + 1.0)
            start = ctx.get(s, np.zeros(self.nf))
            e = np.empty_like(f)
            for j in range(self.nf):
                e[:, j] = ema_run(f[:, j], a, start[j])
            parts.append(e)
            new_ctx[s] = e[-1].copy()
        X = np.concatenate(parts + [np.ones((T, 1))], axis=1)
        return X, (new_ctx if advance else ctx)

    # -- inference -------------------------------------------------------------
    def predict(self, feats: np.ndarray, carry: Any = None):
        f = np.asarray(feats)
        if f.ndim == 3:
            mus, sgs, c = [], [], carry
            for b in range(f.shape[0]):
                m, s, c2 = self.predict(f[b], carry)
                mus.append(m)
                sgs.append(s)
                c = c2 if b == 0 else c
            return np.stack(mus), np.stack(sgs), c
        X, carry2 = self._expand(f, carry, advance=True)
        hpre = X @ self.W1 + self.b1
        h = np.tanh(hpre)
        o = h @ self.W2 + self.b2
        mu = o[:, 0]
        sigma = np.exp(np.clip(o[:, 1], -4, 4))
        if f.ndim == 1:
            return mu[0], sigma[0], carry2
        return mu.astype(np.float32), sigma.astype(np.float32), carry2

    # -- training ----------------------------------------------------------------
    def learn(self, feats: np.ndarray, target: np.ndarray, carry: Any = None,
              weights: Optional[np.ndarray] = None,
              grad_clip: float = 1.0) -> Dict[str, float]:
        c = self.loss_cfg
        X, seam, _ = self._expand_batch(feats, carry, advance=False)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        T = X.shape[0]
        if y.shape[0] != T:                       # pragma: no cover
            raise ValueError(f"target length {y.shape[0]} != features {T}")
        w = (np.asarray(weights, np.float64).reshape(-1) if weights is not None
             else np.ones(T))
        rw = 1.0 + c.reward_weight * (np.abs(y) / (c.move_floor + np.abs(y).mean() + 1e-6))
        w = w * rw

        hpre = X @ self.W1 + self.b1
        h = np.tanh(hpre)
        o = h @ self.W2 + self.b2
        mu, ls = o[:, 0], np.clip(o[:, 1], -4, 4)
        err = mu - y

        a = np.abs(err)
        huber = np.where(a <= c.huber_delta, 0.5 * err ** 2,
                         c.huber_delta * (a - 0.5 * c.huber_delta))
        dhuber = np.where(a <= c.huber_delta, err, c.huber_delta * np.sign(err))
        acc_loss = float((huber * w).mean())
        g_mu = (dhuber * w) / T

        inv = np.exp(-2.0 * ls)
        nll = 0.5 * err ** 2 * inv + ls
        nll_loss = float(nll.mean())
        g_mu += c.nll_weight * (err * inv) / T
        g_ls = c.nll_weight * (-err ** 2 * inv + 1.0) / T

        # smoothness, skipping the seams between concatenated batch elements
        keep = ~seam[1:] if T > 1 else np.zeros(0, dtype=bool)
        d1 = (mu[1:] - mu[:-1]) * keep
        n_keep = max(1, int(keep.sum()))
        smooth = float((d1 ** 2).sum() / n_keep) if T > 1 else 0.0
        if T > 1:
            gs = np.zeros(T)
            gs[1:] += 2 * d1 / n_keep
            gs[:-1] -= 2 * d1 / n_keep
            g_mu += c.smooth_weight * gs

        go = np.stack([g_mu, g_ls], axis=1)
        gW2 = h.T @ go
        gb2 = go.sum(axis=0)
        gh = go @ self.W2.T
        ghpre = gh * (1.0 - h ** 2)
        gW1 = X.T @ ghpre
        gb1 = ghpre.sum(axis=0)

        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
        gn = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        if gn > grad_clip:
            k = grad_clip / (gn + 1e-12)
            grads = {kk: v * k for kk, v in grads.items()}

        self.steps += 1
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        for k, p in self._params().items():
            self._m[k] = b1_ * self._m[k] + (1 - b1_) * grads[k]
            self._v[k] = b2_ * self._v[k] + (1 - b2_) * grads[k] ** 2
            mhat = self._m[k] / (1 - b1_ ** self.steps)
            vhat = self._v[k] / (1 - b2_ ** self.steps)
            p -= self.lr * mhat / (np.sqrt(vhat) + eps)

        big = np.abs(y) > c.move_floor
        dir_acc = float(((mu > 0) == (y > 0))[big].mean()) if big.any() else 0.5
        return {"loss": acc_loss + c.nll_weight * nll_loss + c.smooth_weight * smooth,
                "acc_loss": acc_loss, "nll": nll_loss, "dir_loss": 0.0,
                "smooth": smooth, "curve": 0.0,
                "mae_permille": float(np.abs(err).mean()),
                "dir_acc": dir_acc, "grad_norm": gn, "lr": self.lr}

    def state_dict(self) -> dict:
        return {"backend": "numpy", "steps": self.steps,
                **{k: v.copy() for k, v in self._params().items()}}

    def load_state_dict(self, d: dict) -> None:
        for k in ("W1", "b1", "W2", "b2"):
            getattr(self, k)[...] = np.asarray(d[k])
        self.steps = int(d.get("steps", 0))


# ==========================================================================
def build_predictor(n_features: int, cfg, device: Optional[str] = None,
                    backend: str = "auto", ddp: bool = False,
                    lr: float = 2e-3,
                    loss_cfg: Optional[LossConfig] = None) -> BasePredictor:
    """Pick the best available backend for this machine."""
    if backend == "auto":
        backend = "torch" if _HAS_TORCH else "numpy"
    if backend == "torch":
        if not _HAS_TORCH:
            raise RuntimeError("backend='torch' requested but torch is missing")
        return TorchPredictor(n_features, cfg, device=device or cfg.device,
                              ddp=ddp, amp_dtype=cfg.amp_dtype, lr=lr,
                              loss_cfg=loss_cfg)
    return NumpyPredictor(n_features, cfg, lr=lr, loss_cfg=loss_cfg)


# ==========================================================================
def returns_to_price(close_now: np.ndarray, mu_permille: np.ndarray) -> np.ndarray:
    """Convert the model's per-mille log-return head into an absolute price."""
    return np.asarray(close_now, np.float64) * np.exp(
        np.asarray(mu_permille, np.float64) / RET_SCALE)


def price_to_returns(close_now: np.ndarray, close_future: np.ndarray) -> np.ndarray:
    """Ground-truth target: per-mille log return realised over the horizon."""
    a = np.maximum(np.asarray(close_now, np.float64), 1e-12)
    b = np.maximum(np.asarray(close_future, np.float64), 1e-12)
    return (np.log(b) - np.log(a)) * RET_SCALE
