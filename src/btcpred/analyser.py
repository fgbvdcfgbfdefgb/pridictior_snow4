"""
analyser.py -- the **Market Analyser** (CPU component).

It turns the raw one-second tape into a compact, scale-free signal vector that
the GPU-side Price Predictor consumes.  Everything here is:

* **causal**      -- feature at second *t* uses only data up to and including *t*;
* **stateful**    -- all moving statistics are O(1) recursions, so replaying a
                     month costs the same per second as replaying one minute;
* **exact in both modes** -- ``update_tick()`` (one second) and
  ``update_block()`` (a whole 30-minute window at once) run *the same*
  recursions and produce bit-comparable output.  That is what makes "train on
  whole windows at max speed" equivalent to "update every second".

No dependency beyond NumPy, because the Snowflake side may be offline and
minimal.  The EMA recursion is vectorised exactly (no scipy, no Python loop per
sample) using a numerically-guarded block transform.

Feature groups
--------------
trend      log-price distance from EMAs at 5s ... 2h, MACD, trend slope
momentum   log returns at 1s ... 30m, acceleration, RSI
volatility EMA realised vol at 1m/5m/30m, high-low range, vol-of-vol
flow       volume & trade-count surprise, taker-buy imbalance, VWAP deviation
micro      data-gap density, tick activity
clock      intraday and weekly seasonality (sin/cos)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .compact import Bars

# --------------------------------------------------------------------------
# configuration (frozen -- changing these invalidates checkpoints)
# --------------------------------------------------------------------------
EMA_SPANS = (5, 15, 60, 300, 900, 1800, 7200)
RET_LAGS = (1, 5, 15, 60, 300, 900, 1800)
VOL_SPANS = (60, 300, 1800)
FLOW_SPANS = (60, 900)
RSI_SPAN = 300
MAX_LAG = max(RET_LAGS)

_EPS = 1e-12


def _alpha(span: int) -> float:
    return 2.0 / (span + 1.0)


def ema_run(x: np.ndarray, alpha: float, y0: float) -> np.ndarray:
    """
    Exact EMA  y_t = (1-a) y_{t-1} + a x_t  for a whole array, vectorised.

    Uses the closed form
        y_t = b^t y_0 + a b^t sum_{j<=t} x_j b^(-j),      b = 1 - a
    evaluated in blocks small enough that ``b^-m`` cannot overflow.  This is
    bit-for-bit the same recursion a per-sample loop would produce (up to
    float rounding), but runs at NumPy speed.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return x.copy()
    b = 1.0 - alpha
    if b <= 0.0:
        return x.copy()
    if b >= 1.0:
        return np.full(n, y0, dtype=np.float64)
    # keep b^-m below ~1e250
    m_max = int(max(1.0, 250.0 * math.log(10.0) / (-math.log(b))))
    out = np.empty(n, dtype=np.float64)
    prev = float(y0)
    for s in range(0, n, m_max):
        xb = x[s:s + m_max]
        m = xb.shape[0]
        k = np.arange(1, m + 1, dtype=np.float64)
        bk = np.power(b, k)                 # b^1 .. b^m
        c = xb / bk                         # x_j * b^-j
        out[s:s + m] = bk * prev + alpha * bk * np.cumsum(c)
        prev = out[s + m - 1]
    return out


# --------------------------------------------------------------------------
def _build_names() -> List[str]:
    n: List[str] = []
    n += [f"trend_ema{s}" for s in EMA_SPANS]
    n += ["trend_macd", "trend_macd_sig", "trend_macd_hist", "trend_slope"]
    n += [f"mom_ret{l}" for l in RET_LAGS]
    n += ["mom_accel", "mom_rsi"]
    n += [f"vol_rv{s}" for s in VOL_SPANS]
    n += ["vol_range", "vol_range_ema", "vol_of_vol"]
    n += [f"flow_vol_z{s}" for s in FLOW_SPANS]
    n += [f"flow_trades_z{s}" for s in FLOW_SPANS]
    n += ["flow_imbalance_fast", "flow_imbalance_slow", "flow_vwap_dev",
          "flow_notional"]
    n += ["micro_gap_density", "micro_band_pos"]
    n += ["clock_tod_sin", "clock_tod_cos", "clock_dow_sin", "clock_dow_cos"]
    return n


FEATURE_NAMES: List[str] = _build_names()
N_FEATURES: int = len(FEATURE_NAMES)


# --------------------------------------------------------------------------
@dataclass
class AnalyserState:
    """Everything the recursions need to continue across a block boundary."""
    initialised: bool = False
    ema: Dict[int, float] = field(default_factory=dict)        # log-price EMAs
    macd_sig: float = 0.0
    rv: Dict[int, float] = field(default_factory=dict)         # EMA of r^2
    vov: float = 0.0
    rng_ema: float = 0.0
    vol_ema: Dict[int, float] = field(default_factory=dict)
    trd_ema: Dict[int, float] = field(default_factory=dict)
    imb_fast: float = 0.5
    imb_slow: float = 0.5
    gap_ema: float = 0.0
    hi_ema: float = 0.0
    lo_ema: float = 0.0
    up_ema: float = 0.0
    dn_ema: float = 0.0
    last_r: float = 0.0
    logp_hist: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_LAG + 1, dtype=np.float64))
    n_seen: int = 0

    def clone(self) -> "AnalyserState":
        import copy
        return copy.deepcopy(self)


class MarketAnalyser:
    """
    CPU feature/signal extractor.

    >>> an = MarketAnalyser()
    >>> an.warmup(history_bars)            # prime the long EMAs
    >>> feats = an.update_block(bars)      # (N, N_FEATURES) float32
    >>> one   = an.update_tick(tick)       # (N_FEATURES,)   float32
    """

    n_features = N_FEATURES
    feature_names = FEATURE_NAMES

    def __init__(self, state: Optional[AnalyserState] = None):
        self.state = state or AnalyserState()

    # -- helpers -----------------------------------------------------------
    def _init_state(self, logp0: float, bars: Bars) -> None:
        s = self.state
        for sp in EMA_SPANS:
            s.ema[sp] = logp0
        for sp in VOL_SPANS:
            s.rv[sp] = 0.0
        for sp in FLOW_SPANS:
            s.vol_ema[sp] = float(max(bars.volume[0], _EPS))
            s.trd_ema[sp] = float(max(bars.trades[0], 1.0))
        s.macd_sig = 0.0
        s.hi_ema = float(bars.high[0])
        s.lo_ema = float(bars.low[0])
        s.rng_ema = 0.0
        s.vov = 0.0
        s.up_ema = s.dn_ema = _EPS
        s.last_r = 0.0
        s.logp_hist[:] = logp0
        s.initialised = True

    # -- main entry point --------------------------------------------------
    def update_block(self, bars: Bars, advance: bool = True) -> np.ndarray:
        """
        Compute features for every second of ``bars`` and advance the state.

        Returns ``(len(bars), N_FEATURES)`` float32.
        """
        n = len(bars)
        if n == 0:
            return np.zeros((0, N_FEATURES), np.float32)
        s = self.state if advance else self.state.clone()

        close = np.asarray(bars.close, dtype=np.float64)
        high = np.asarray(bars.high, dtype=np.float64)
        low = np.asarray(bars.low, dtype=np.float64)
        vol = np.asarray(bars.volume, dtype=np.float64)
        qvol = np.asarray(bars.quote_volume, dtype=np.float64)
        trd = np.asarray(bars.trades, dtype=np.float64)
        tbb = np.asarray(bars.taker_buy_base, dtype=np.float64)
        gap = np.asarray(bars.gap, dtype=np.float64)

        logp = np.log(np.maximum(close, _EPS))
        if not s.initialised:
            self._init_state(float(logp[0]), bars)

        cols: List[np.ndarray] = []

        # ---------- trend --------------------------------------------------
        emas: Dict[int, np.ndarray] = {}
        for sp in EMA_SPANS:
            e = ema_run(logp, _alpha(sp), s.ema[sp])
            emas[sp] = e
            cols.append((logp - e) * 1e3)                # log distance, "per-mille"
        macd = (emas[60] - emas[300]) * 1e3
        macd_sig = ema_run(macd, _alpha(60), s.macd_sig)
        cols += [macd, macd_sig, macd - macd_sig]
        cols.append((emas[300] - emas[1800]) * 1e3)      # slope proxy

        # ---------- momentum ------------------------------------------------
        hist = s.logp_hist                                # last MAX_LAG+1 values
        ext = np.concatenate([hist[1:], logp])            # continuous history
        base = hist.shape[0] - 1                          # index of logp[0]-1 in ext
        for lag in RET_LAGS:
            prev = ext[base + 1 - lag: base + 1 - lag + n]
            cols.append((logp - prev) * 1e3)
        r1 = cols[len(EMA_SPANS) + 4]                     # mom_ret1 (per-mille)
        accel = np.empty(n)
        accel[0] = r1[0] - s.last_r
        accel[1:] = r1[1:] - r1[:-1]
        cols.append(accel)

        up = ema_run(np.maximum(r1, 0.0), _alpha(RSI_SPAN), s.up_ema)
        dn = ema_run(np.maximum(-r1, 0.0), _alpha(RSI_SPAN), s.dn_ema)
        rsi = up / (up + dn + _EPS)                       # 0..1
        cols.append(rsi * 2.0 - 1.0)

        # ---------- volatility ------------------------------------------------
        r2 = r1 * r1
        rvs = {}
        for sp in VOL_SPANS:
            e = ema_run(r2, _alpha(sp), s.rv[sp])
            rvs[sp] = e
            cols.append(np.sqrt(np.maximum(e, 0.0)))
        rng = (high - low) / np.maximum(close, _EPS) * 1e3
        rng_e = ema_run(rng, _alpha(300), s.rng_ema)
        cols += [rng, rng_e]
        vov = ema_run(np.abs(rvs[60] - rvs[300]), _alpha(900), s.vov)
        cols.append(vov)

        # ---------- flow --------------------------------------------------------
        vol_es, trd_es = {}, {}
        for sp in FLOW_SPANS:
            e = ema_run(vol, _alpha(sp), s.vol_ema[sp])
            vol_es[sp] = e
            cols.append(np.log1p(vol) - np.log1p(np.maximum(e, 0.0)))
        for sp in FLOW_SPANS:
            e = ema_run(trd, _alpha(sp), s.trd_ema[sp])
            trd_es[sp] = e
            cols.append(np.log1p(trd) - np.log1p(np.maximum(e, 0.0)))
        ratio = np.where(vol > 0, tbb / np.maximum(vol, _EPS), 0.5)
        imb_f = ema_run(ratio, _alpha(60), s.imb_fast)
        imb_s = ema_run(ratio, _alpha(300), s.imb_slow)
        cols += [imb_f * 2.0 - 1.0, imb_s * 2.0 - 1.0]
        vwap = np.where(vol > 0, qvol / np.maximum(vol, _EPS), close)
        cols.append((vwap - close) / np.maximum(close, _EPS) * 1e3)
        cols.append(np.log1p(qvol) - 10.0)                # notional, de-meaned

        # ---------- micro / regime ----------------------------------------------
        gap_e = ema_run(gap, _alpha(300), s.gap_ema)
        cols.append(gap_e)
        hi_e = ema_run(high, _alpha(900), s.hi_ema)
        lo_e = ema_run(low, _alpha(900), s.lo_ema)
        band = np.maximum(hi_e - lo_e, _EPS)
        cols.append(np.clip((close - lo_e) / band, -3.0, 4.0) - 0.5)

        # ---------- clock ---------------------------------------------------------
        ts = np.arange(bars.t0, bars.t0 + n, dtype=np.float64)
        tod = (ts % 86400.0) / 86400.0 * (2 * math.pi)
        dow = (ts % 604800.0) / 604800.0 * (2 * math.pi)
        cols += [np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow)]

        # ---------- commit state ---------------------------------------------------
        if advance:
            for sp in EMA_SPANS:
                s.ema[sp] = float(emas[sp][-1])
            s.macd_sig = float(macd_sig[-1])
            for sp in VOL_SPANS:
                s.rv[sp] = float(rvs[sp][-1])
            s.vov = float(vov[-1])
            s.rng_ema = float(rng_e[-1])
            for sp in FLOW_SPANS:
                s.vol_ema[sp] = float(vol_es[sp][-1])
                s.trd_ema[sp] = float(trd_es[sp][-1])
            s.imb_fast = float(imb_f[-1])
            s.imb_slow = float(imb_s[-1])
            s.gap_ema = float(gap_e[-1])
            s.hi_ema = float(hi_e[-1])
            s.lo_ema = float(lo_e[-1])
            s.up_ema = float(up[-1])
            s.dn_ema = float(dn[-1])
            s.last_r = float(r1[-1])
            s.logp_hist = ext[-(MAX_LAG + 1):].copy()
            s.n_seen += n

        out = np.stack(cols, axis=1)
        if out.shape[1] != N_FEATURES:       # pragma: no cover -- guards edits
            raise RuntimeError(f"feature count drift: {out.shape[1]} != {N_FEATURES}")
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0
                             ).astype(np.float32, copy=False)

    # -- single second ------------------------------------------------------
    def update_tick(self, tick) -> np.ndarray:
        """Feature vector for exactly one second (same math as ``update_block``)."""
        b = Bars(
            t0=int(tick.ts),
            close=np.array([tick.close], np.float32),
            open=np.array([tick.open], np.float32),
            high=np.array([tick.high], np.float32),
            low=np.array([tick.low], np.float32),
            volume=np.array([tick.volume], np.float32),
            quote_volume=np.array([tick.quote_volume], np.float32),
            trades=np.array([tick.trades], np.int32),
            taker_buy_base=np.array([tick.taker_buy_base], np.float32),
            gap=np.array([tick.gap], bool),
        )
        return self.update_block(b)[0]

    # -- warm-up ------------------------------------------------------------
    def warmup(self, bars: Bars) -> None:
        """Run the recursions over history without returning anything."""
        if len(bars):
            self.update_block(bars)

    # -- persistence --------------------------------------------------------
    def state_dict(self) -> dict:
        s = self.state
        return {
            "initialised": s.initialised, "ema": s.ema, "macd_sig": s.macd_sig,
            "rv": s.rv, "vov": s.vov, "rng_ema": s.rng_ema,
            "vol_ema": s.vol_ema, "trd_ema": s.trd_ema,
            "imb_fast": s.imb_fast, "imb_slow": s.imb_slow, "gap_ema": s.gap_ema,
            "hi_ema": s.hi_ema, "lo_ema": s.lo_ema, "up_ema": s.up_ema,
            "dn_ema": s.dn_ema, "last_r": s.last_r,
            "logp_hist": s.logp_hist.tolist(), "n_seen": s.n_seen,
        }

    def load_state_dict(self, d: dict) -> None:
        s = self.state
        s.initialised = bool(d["initialised"])
        s.ema = {int(k): float(v) for k, v in d["ema"].items()}
        s.rv = {int(k): float(v) for k, v in d["rv"].items()}
        s.vol_ema = {int(k): float(v) for k, v in d["vol_ema"].items()}
        s.trd_ema = {int(k): float(v) for k, v in d["trd_ema"].items()}
        for k in ("macd_sig", "vov", "rng_ema", "imb_fast", "imb_slow",
                  "gap_ema", "hi_ema", "lo_ema", "up_ema", "dn_ema", "last_r"):
            setattr(s, k, float(d[k]))
        s.logp_hist = np.asarray(d["logp_hist"], dtype=np.float64)
        s.n_seen = int(d["n_seen"])


# --------------------------------------------------------------------------
# normalisation -- features are already scale-free, this just tames outliers
# --------------------------------------------------------------------------
class RunningNormaliser:
    """Welford mean/var with clipping; shared across ranks via ``sync_from``."""

    def __init__(self, n: int = N_FEATURES, clip: float = 8.0, momentum: float = 1e-4):
        self.mean = np.zeros(n, np.float64)
        self.var = np.ones(n, np.float64)
        self.clip = float(clip)
        self.momentum = float(momentum)
        self.count = 0

    def observe(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.mean.shape[0])
        if x.size == 0:
            return
        m = x.mean(axis=0)
        v = x.var(axis=0)
        k = min(1.0, self.momentum * x.shape[0])
        self.mean = (1 - k) * self.mean + k * m
        self.var = (1 - k) * self.var + k * np.maximum(v, 1e-8)
        self.count += x.shape[0]

    def apply(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, np.float32) - self.mean.astype(np.float32)) / \
            np.sqrt(self.var.astype(np.float32) + 1e-6)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(),
                "count": self.count, "clip": self.clip, "momentum": self.momentum}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], np.float64)
        self.var = np.asarray(d["var"], np.float64)
        self.count = int(d.get("count", 0))
        self.clip = float(d.get("clip", 8.0))
        self.momentum = float(d.get("momentum", 1e-4))
