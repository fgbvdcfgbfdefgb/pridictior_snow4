"""
metrics.py -- accuracy, reward and stability accounting.

Everything here operates on a *window* (1800 per-second predictions and the
1800 prices that were actually printed 30 minutes later).  The numbers feed
three consumers:

* the training loop (the reward signal fed back into the next cycle),
* the live visualisation (the accuracy progress bars),
* the checkpoint record (so any run can be audited after the fact).

"Accuracy" for a price forecaster is ambiguous, so four complementary
definitions are reported rather than one misleading number:

``band_accuracy``   fraction of seconds whose predicted price lands within a
                    tolerance band of the real one (default 10 bps).  This is
                    the headline number the progress bars show.
``direction_acc``   fraction of *material* moves whose sign was called right.
``r2``              coefficient of determination against the realised return --
                    harsh, and negative until the model genuinely beats "flat".
``skill``           improvement over the naive random-walk forecast
                    ("price in 30 minutes == price now"), which is the only
                    benchmark that matters for second-by-second BTC.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

RET_SCALE = 1000.0


# --------------------------------------------------------------------------
@dataclass
class WindowMetrics:
    n: int
    mae_permille: float
    rmse_permille: float
    mape_pct: float
    band_accuracy: float
    direction_acc: float
    r2: float
    skill_vs_naive: float
    reward: float
    bias_permille: float
    jitter_permille: float
    max_slew_permille: float
    sigma_mean: float
    calibration: float

    def to_dict(self) -> dict:
        return {k: (float(v) if not isinstance(v, int) else v)
                for k, v in asdict(self).items()}


def evaluate_window(pred_permille: np.ndarray,
                    true_permille: np.ndarray,
                    price_now: np.ndarray,
                    sigma: Optional[np.ndarray] = None,
                    band_bps: float = 10.0,
                    move_floor_permille: float = 0.25) -> WindowMetrics:
    """
    Score one window of per-second forecasts.

    ``pred_permille``  model output, 1000 * ln(P_hat[t+H] / P[t])
    ``true_permille``  realised,     1000 * ln(P[t+H]     / P[t])
    ``price_now``      P[t], used for the price-space band accuracy
    """
    p = np.asarray(pred_permille, np.float64).reshape(-1)
    y = np.asarray(true_permille, np.float64).reshape(-1)
    c = np.asarray(price_now, np.float64).reshape(-1)
    n = min(p.size, y.size, c.size)
    p, y, c = p[:n], y[:n], c[:n]
    if n == 0:
        return WindowMetrics(0, *([0.0] * 13))

    err = p - y
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))

    pred_price = c * np.exp(p / RET_SCALE)
    true_price = c * np.exp(y / RET_SCALE)
    rel = np.abs(pred_price - true_price) / np.maximum(true_price, 1e-9)
    mape = float(rel.mean() * 100.0)
    band = float((rel <= band_bps / 10_000.0).mean())

    big = np.abs(y) > move_floor_permille
    dir_acc = float(((p > 0) == (y > 0))[big].mean()) if big.any() else 0.5

    ss_res = float((err ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # naive forecast = "no change", i.e. prediction 0
    naive = float((y ** 2).mean())
    skill = float(1.0 - (err ** 2).mean() / naive) if naive > 0 else 0.0

    # trading-flavoured reward: did we capture the move we called?
    reward = float(np.mean(np.sign(p) * y * (np.abs(y) > move_floor_permille)))

    d = np.diff(p) if n > 1 else np.zeros(1)
    sg = (np.asarray(sigma, np.float64).reshape(-1)[:n] if sigma is not None
          else np.ones(n))
    z = np.abs(err) / np.maximum(sg, 1e-6)
    calib = float((z <= 1.0).mean())          # ideal ~0.68 for a Gaussian

    return WindowMetrics(
        n=int(n), mae_permille=mae, rmse_permille=rmse, mape_pct=mape,
        band_accuracy=band, direction_acc=dir_acc, r2=r2,
        skill_vs_naive=skill, reward=reward,
        bias_permille=float(err.mean()),
        jitter_permille=float(np.abs(d).mean()),
        max_slew_permille=float(np.abs(d).max()) if d.size else 0.0,
        sigma_mean=float(sg.mean()), calibration=calib,
    )


# --------------------------------------------------------------------------
class RunningMetrics:
    """EMA of window metrics -- what the progress bars display."""

    def __init__(self, span: int = 20):
        self.alpha = 2.0 / (span + 1.0)
        self.values: Dict[str, float] = {}
        self.best: Dict[str, float] = {}
        self.n = 0

    def update(self, m: WindowMetrics | Dict[str, float]) -> Dict[str, float]:
        d = m.to_dict() if isinstance(m, WindowMetrics) else dict(m)
        for k, v in d.items():
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                continue
            self.values[k] = v if k not in self.values else \
                (1 - self.alpha) * self.values[k] + self.alpha * v
            if k in ("band_accuracy", "direction_acc", "skill_vs_naive", "r2", "reward"):
                self.best[k] = max(self.best.get(k, -1e18), self.values[k])
            elif k in ("mae_permille", "rmse_permille", "jitter_permille", "mape_pct"):
                self.best[k] = min(self.best.get(k, 1e18), self.values[k])
        self.n += 1
        return self.values

    def get(self, k: str, default: float = 0.0) -> float:
        return float(self.values.get(k, default))

    def snapshot(self) -> dict:
        return {"ema": dict(self.values), "best": dict(self.best), "windows": self.n}


# --------------------------------------------------------------------------
def progress_bar(value: float, lo: float = 0.0, hi: float = 1.0,
                 width: int = 24, fill: str = "#", empty: str = "-") -> str:
    """Text progress bar used by the console trainer."""
    if not np.isfinite(value):
        value = lo
    f = (float(value) - lo) / max(1e-12, hi - lo)
    f = min(1.0, max(0.0, f))
    k = int(round(f * width))
    return f"[{fill * k}{empty * (width - k)}]"


ACCURACY_BARS = (
    # key,              label,            lo,   hi,   fmt
    ("band_accuracy",   "band 10bps",     0.0,  1.0,  "{:.1%}"),
    ("direction_acc",   "direction",      0.3,  0.8,  "{:.1%}"),
    ("skill_vs_naive",  "skill vs naive", -0.5, 0.5,  "{:+.3f}"),
    ("calibration",     "calibration",    0.0,  1.0,  "{:.1%}"),
)


def render_bars(values: Dict[str, float], width: int = 24) -> str:
    out = []
    for key, label, lo, hi, fmt in ACCURACY_BARS:
        v = float(values.get(key, lo))
        out.append(f"  {label:<16s}{progress_bar(v, lo, hi, width)} {fmt.format(v)}")
    return "\n".join(out)


def stability_report(pred: np.ndarray, limit_permille: float = 0.08) -> Dict[str, float]:
    p = np.asarray(pred, np.float64).reshape(-1)
    if p.size < 2:
        return {"jitter": 0.0, "max_slew": 0.0, "breach_frac": 0.0, "n": int(p.size)}
    d = np.abs(np.diff(p))
    return {"jitter": float(d.mean()), "max_slew": float(d.max()),
            "breach_frac": float((d > limit_permille).mean()), "n": int(p.size)}
