"""
stability.py -- keep the published forecast calm enough to trade on.

A raw per-second forecast is useless to a trading system if it flickers: a
position sized off a number that jumps 0.4 % between two consecutive seconds
will churn itself to death in fees.  The brief is explicit -- *predictions must
be stable (no erratic jumps)*.

Stability is enforced at three levels:

1.  **In the loss** (``model.LossConfig.smooth_weight`` / ``smooth2_weight``)
    -- the network is penalised for first and second differences of its own
    output, so it learns to be smooth instead of being smoothed.
2.  **At inference, here** -- a causal three-stage filter:
      a) confidence-weighted EMA (wide predicted sigma => slower adaptation),
      b) hard slew-rate limit (max basis points of change per second),
      c) a dead-band that ignores micro-revisions below the noise floor.
3.  **In the metrics** (``metrics.stability_report``) -- jitter, max slew and
    the fraction of seconds that breached the limiter are tracked every window
    and checkpointed, so regressions are visible rather than silent.

Every filter here is causal and O(1); the streaming and batch paths are the
same recursion, so the live animation and the offline run agree exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class StabilityConfig:
    ema_span: float = 45.0          # seconds; base smoothing of the forecast
    conf_sensitivity: float = 0.8   # how much predicted sigma slows adaptation
    max_slew_permille: float = 0.08 # max change of the forecast per second
    dead_band_permille: float = 0.01
    warmup_seconds: int = 60        # trust the raw signal less while warming up
    max_abs_permille: float = 120.0 # sanity clamp (12 % over 30 min)

    def to_dict(self) -> dict:
        return asdict(self)


class StabilityFilter:
    """
    Causal smoother for the per-second forecast stream.

    >>> f = StabilityFilter(StabilityConfig())
    >>> smooth, info = f.apply_block(mu, sigma)     # whole window
    >>> one = f.apply_one(mu_t, sigma_t)            # single second
    """

    def __init__(self, cfg: Optional[StabilityConfig] = None):
        self.cfg = cfg or StabilityConfig()
        self.y: Optional[float] = None
        self.n = 0
        self.clipped = 0
        self.total = 0

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        self.y = None
        self.n = 0

    def state_dict(self) -> dict:
        return {"y": self.y, "n": self.n, "cfg": self.cfg.to_dict()}

    def load_state_dict(self, d: dict) -> None:
        self.y = d.get("y")
        self.n = int(d.get("n", 0))
        if "cfg" in d:
            self.cfg = StabilityConfig(**d["cfg"])

    # -- core ---------------------------------------------------------------
    def _alpha(self, sigma: float) -> float:
        """Confidence-weighted smoothing factor; wide sigma => smaller alpha."""
        c = self.cfg
        base = 2.0 / (c.ema_span + 1.0)
        if self.n < c.warmup_seconds:                 # converge faster at start
            base = min(0.5, base * (1.0 + 4.0 * (1.0 - self.n / max(1, c.warmup_seconds))))
        conf = 1.0 / (1.0 + c.conf_sensitivity * max(0.0, float(sigma)))
        return float(np.clip(base * conf, 1e-4, 0.95))

    def apply_one(self, mu: float, sigma: float = 1.0) -> float:
        mu = float(np.clip(mu, -self.cfg.max_abs_permille, self.cfg.max_abs_permille))
        if self.y is None:
            self.y = mu
            self.n = 1
            self.total += 1
            return mu
        a = self._alpha(sigma)
        cand = (1.0 - a) * self.y + a * mu
        step = cand - self.y
        lim = self.cfg.max_slew_permille
        self.total += 1
        if abs(step) > lim:
            self.clipped += 1
            step = np.sign(step) * lim
        if abs(step) < self.cfg.dead_band_permille:
            step = 0.0
        self.y = float(self.y + step)
        self.n += 1
        return self.y

    def apply_block(self, mu: np.ndarray, sigma: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Filter a whole window.  Identical to calling ``apply_one`` in a loop
        (the recursion is inherently sequential -- 1800 iterations per window
        is nothing compared with the forward pass).
        """
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        sg = (np.asarray(sigma, dtype=np.float64).reshape(-1)
              if sigma is not None else np.ones_like(mu))
        out = np.empty_like(mu)
        c0 = self.clipped
        for i in range(mu.shape[0]):
            out[i] = self.apply_one(mu[i], sg[i])
        d = np.diff(out) if out.size > 1 else np.zeros(1)
        info = {
            "jitter_permille": float(np.abs(d).mean()),
            "max_slew_permille": float(np.abs(d).max()) if d.size else 0.0,
            "clipped_frac": float((self.clipped - c0) / max(1, mu.shape[0])),
            "raw_jitter_permille": float(np.abs(np.diff(mu)).mean()) if mu.size > 1 else 0.0,
        }
        return out, info


# --------------------------------------------------------------------------
def smoothness_gain(raw: np.ndarray, smoothed: np.ndarray) -> float:
    """How much twitchiness the filter removed (1.0 = none left)."""
    r = np.abs(np.diff(np.asarray(raw, np.float64)))
    s = np.abs(np.diff(np.asarray(smoothed, np.float64)))
    if r.size == 0 or r.mean() <= 0:
        return 0.0
    return float(1.0 - s.mean() / r.mean())
