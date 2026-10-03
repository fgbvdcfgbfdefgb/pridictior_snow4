"""
viz.py -- real-time visualisation of the forecast.

Three renderers over one data model, so the picture is the same everywhere:

``animate_run()``        matplotlib -> animated GIF / MP4 of a finished run
``LiveChart``            in-process / in-notebook live redraw (Snowflake uses
                         this; it clears and redraws the same figure as the
                         simulator streams)
``build_standalone_html()``  a single self-contained .html file with the data
                         embedded and a hand-written canvas animation -- no
                         CDN, no network, so it plays inside an air-gapped
                         Snowflake notebook, in an email attachment, or in a
                         sandboxed preview pane

What is drawn
-------------
* the **actual** market price, solid;
* the model's **predicted** price, dotted, plotted at the time it is a forecast
  *for* (i.e. shifted +30 min), so you read it as "at this moment the model
  thought the price here would be X";
* a shaded +/-sigma confidence ribbon;
* **training accuracy progress bars** (band accuracy, direction, skill vs the
  naive random walk, calibration);
* throughput and loss tickers.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HORIZON = 1800
BAR_SPECS = (
    ("band_accuracy", "band 10bps", 0.0, 1.0, "{:.1%}"),
    ("direction_acc", "direction", 0.3, 0.8, "{:.1%}"),
    ("skill_vs_naive", "skill vs naive", -0.5, 0.5, "{:+.3f}"),
    ("calibration", "calibration", 0.0, 1.0, "{:.1%}"),
)
COL_BG = "#0e1117"
COL_FG = "#e6edf3"
COL_GRID = "#262c36"
COL_PRICE = "#4cc9f0"
COL_PRED = "#ffb703"
COL_BAND = "#ffb70333"
COL_OK = "#3fb950"
COL_WARN = "#f0883e"


# ==========================================================================
@dataclass
class RunSeries:
    """A run's per-second tape, flattened and time-ordered."""
    ts: np.ndarray
    price: np.ndarray
    pred_price: np.ndarray
    sigma_price: np.ndarray
    target_price: np.ndarray
    window: np.ndarray

    def __len__(self) -> int:
        return int(self.ts.shape[0])


def load_run_series(run_dir: str, lane: int = 0, rank: int = 0,
                    max_windows: Optional[int] = None) -> RunSeries:
    """Concatenate the per-window prediction files of one lane into one tape."""
    pred_dir = os.path.join(run_dir, "predictions")
    suffix = "" if rank == 0 else f"_r{rank}"
    files = sorted(glob.glob(os.path.join(pred_dir, f"w*{suffix}.npz")))
    # window ids are  window_index * 1000 + lane_id
    keep = []
    for f in files:
        stem = os.path.basename(f).split(".")[0].replace("w", "").split("_")[0]
        try:
            if int(stem) % 1000 == lane:
                keep.append(f)
        except ValueError:
            continue
    if not keep:
        keep = files
    if max_windows:
        keep = keep[-max_windows:]
    if not keep:
        raise FileNotFoundError(f"no prediction windows under {pred_dir}")

    ts, pr, pp, sg, tg, wi = [], [], [], [], [], []
    for f in keep:
        with np.load(f) as z:
            n = int(z["ts"].shape[0])
            ts.append(z["ts"])
            pr.append(z["price_now"])
            pp.append(z["pred_price"])
            s = z["sigma"].astype(np.float64)
            sg.append(z["price_now"].astype(np.float64) *
                      (np.exp(s / 1000.0) - 1.0))
            tg.append(z["true_future_price"] if "true_future_price" in z.files
                      else np.full(n, np.nan, np.float32))
            wi.append(np.full(n, int(z["window"]) // 1000, np.int64))
    order = np.argsort(np.concatenate(ts), kind="stable")
    g = lambda L: np.concatenate(L)[order]
    return RunSeries(ts=g(ts).astype(np.int64), price=g(pr).astype(np.float64),
                     pred_price=g(pp).astype(np.float64),
                     sigma_price=g(sg).astype(np.float64),
                     target_price=g(tg).astype(np.float64),
                     window=g(wi).astype(np.int64))


def load_metrics(run_dir: str) -> List[Dict[str, float]]:
    import csv
    p = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as fh:
        for row in csv.DictReader(fh):
            d = {}
            for k, v in row.items():
                try:
                    d[k] = float(v) if v not in ("", None) else float("nan")
                except (TypeError, ValueError):
                    d[k] = v
            out.append(d)
    return out


# ==========================================================================
def _style(ax, title: Optional[str] = None):
    ax.set_facecolor(COL_BG)
    for s in ax.spines.values():
        s.set_color(COL_GRID)
    ax.tick_params(colors=COL_FG, labelsize=8)
    ax.grid(True, color=COL_GRID, lw=0.6, alpha=0.8)
    if title:
        ax.set_title(title, color=COL_FG, fontsize=10, loc="left")
    return ax


def _draw_bars(ax, values: Dict[str, float],
               title: str = "training accuracy") -> None:
    """Accuracy progress bars."""
    ax.clear()
    _style(ax, title)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(BAR_SPECS) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for i, (key, label, lo, hi, fmt) in enumerate(BAR_SPECS):
        v = float(values.get(key, lo) if np.isfinite(values.get(key, lo)) else lo)
        f = min(1.0, max(0.0, (v - lo) / max(1e-9, hi - lo)))
        y = len(BAR_SPECS) - 1 - i
        ax.barh(y, 1.0, height=0.45, color=COL_GRID, edgecolor="none")
        ax.barh(y, f, height=0.45,
                color=COL_OK if f > 0.5 else COL_WARN, edgecolor="none")
        ax.text(0.01, y + 0.33, label, color=COL_FG, fontsize=8, va="bottom")
        ax.text(0.99, y + 0.33, fmt.format(v), color=COL_FG, fontsize=8,
                va="bottom", ha="right")


def plot_frame(fig, axes, series: RunSeries, i: int, window_s: int = 5400,
               metrics: Optional[Dict[str, float]] = None,
               info: str = "", bars_title: str = "training accuracy") -> None:
    """Render one frame: price + forecast + accuracy bars."""
    ax_p, ax_b, ax_l = axes
    a = max(0, i - window_s)
    ts = series.ts[a:i + 1]
    if ts.size < 2:
        return
    ax_p.clear()
    _style(ax_p, "BTCUSDT  --  actual vs model forecast (30 min ahead)")

    x = (ts - ts[0]) / 60.0
    ax_p.plot(x, series.price[a:i + 1], color=COL_PRICE, lw=1.4,
              label="actual price")
    # the forecast made at t is a statement about t + 30 min
    xf = x + HORIZON / 60.0
    pf = series.pred_price[a:i + 1]
    sf = series.sigma_price[a:i + 1]
    ax_p.plot(xf, pf, color=COL_PRED, lw=1.5, ls=":", label="predicted (t+30m)")
    ax_p.fill_between(xf, pf - sf, pf + sf, color=COL_BAND, lw=0)
    ax_p.axvline(x[-1], color=COL_FG, lw=0.8, alpha=0.5)
    ax_p.text(x[-1], ax_p.get_ylim()[1], "  now", color=COL_FG, fontsize=8,
              va="top")
    ax_p.set_xlabel("minutes", color=COL_FG, fontsize=8)
    ax_p.set_ylabel("USD", color=COL_FG, fontsize=8)
    leg = ax_p.legend(loc="upper left", fontsize=8, facecolor=COL_BG,
                      edgecolor=COL_GRID)
    for t in leg.get_texts():
        t.set_color(COL_FG)
    if info:
        ax_p.text(0.995, 0.02, info, transform=ax_p.transAxes, color=COL_FG,
                  fontsize=8, ha="right", va="bottom", alpha=0.85)

    _draw_bars(ax_b, metrics or {}, title=bars_title)

    ax_l.clear()
    _style(ax_l, "forecast error (per-mille log-return)")
    valid = np.isfinite(series.target_price[a:i + 1])
    if valid.any():
        err = (np.log(np.maximum(series.pred_price[a:i + 1], 1e-9))
               - np.log(np.maximum(series.target_price[a:i + 1], 1e-9))) * 1000.0
        ax_l.plot(x[valid], err[valid], color=COL_WARN, lw=1.0)
        ax_l.axhline(0, color=COL_FG, lw=0.7, alpha=0.5)
    ax_l.set_xlabel("minutes", color=COL_FG, fontsize=8)


def animate_run(run_dir: str, out_path: str = "artifacts/live_prediction.gif",
                lane: int = 0, fps: int = 12, frames: int = 150,
                window_s: int = 5400, stride: Optional[int] = None,
                dpi: int = 100) -> str:
    """Render a finished run to an animated GIF (or MP4 if ffmpeg exists)."""
    bars_title = "training accuracy"
    try:
        with open(os.path.join(run_dir, "run.json")) as fh:
            if json.load(fh).get("kind") == "replay":
                bars_title = "accuracy (held-out replay)"
    except Exception:
        pass
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    series = load_run_series(run_dir, lane=lane)
    mrows = load_metrics(run_dir)
    n = len(series)
    if n < 10:
        raise ValueError(f"run {run_dir} has too little data to animate ({n})")
    start = min(window_s, n - 1)
    stride = stride or max(1, (n - start) // max(1, frames))
    idx = list(range(start, n, stride))[:frames] or [n - 1]

    fig = plt.figure(figsize=(11, 6.2), facecolor=COL_BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1.0], width_ratios=[2.2, 1.0],
                          hspace=0.35, wspace=0.22,
                          left=0.07, right=0.98, top=0.93, bottom=0.09)
    ax_p = fig.add_subplot(gs[0, :])
    ax_l = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])
    axes = (ax_p, ax_b, ax_l)

    def metrics_at(i: int) -> Dict[str, float]:
        if not mrows:
            return {}
        w = int(series.window[i]) if i < len(series.window) else 0
        row = min(mrows, key=lambda r: abs(float(r.get("window", 0)) - w))
        return row

    def frame(i: int):
        m = metrics_at(i)
        import datetime as dt
        utc = dt.datetime.fromtimestamp(int(series.ts[i]), dt.timezone.utc)
        info = (f"{utc:%Y-%m-%d %H:%M:%S} UTC   "
                f"window {int(series.window[i])}   "
                f"{m.get('sim_sec_per_sec', float('nan')):,.0f} sim-s/s")
        plot_frame(fig, axes, series, i, window_s=window_s, metrics=m,
                   info=info, bars_title=bars_title)
        return []

    anim = FuncAnimation(fig, frame, frames=idx, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if out_path.endswith(".mp4"):
        try:
            from matplotlib.animation import FFMpegWriter
            anim.save(out_path, writer=FFMpegWriter(fps=fps), dpi=dpi)
            plt.close(fig)
            return out_path
        except Exception:
            out_path = out_path[:-4] + ".gif"
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path


def _ema(y: np.ndarray, span: int) -> np.ndarray:
    """Smoothing for display only -- per-window metrics are extremely noisy."""
    y = np.asarray(y, np.float64)
    out = np.full_like(y, np.nan)
    a = 2.0 / (span + 1.0)
    acc = None
    for i, v in enumerate(y):
        if not np.isfinite(v):
            out[i] = acc if acc is not None else np.nan
            continue
        acc = v if acc is None else (1 - a) * acc + a * v
        out[i] = acc
    return out


def _robust_ylim(ax, series: Sequence[np.ndarray], lo_q: float = 1.0,
                 hi_q: float = 99.0, pad: float = 0.1) -> None:
    """Ignore the handful of spikes that would otherwise flatten the plot."""
    vals = np.concatenate([s[np.isfinite(s)] for s in series
                           if np.isfinite(s).any()] or [np.array([0.0, 1.0])])
    lo, hi = np.percentile(vals, [lo_q, hi_q])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max()) + 1e-9
    m = (hi - lo) * pad
    ax.set_ylim(lo - m, hi + m)


def plot_summary(run_dir: str, out_path: str = "artifacts/training_curves.png",
                 smooth: int = 40) -> str:
    """Static multi-panel summary of a run (loss, accuracy, error, stability)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_metrics(run_dir)
    if not rows:
        raise FileNotFoundError(f"no metrics.csv in {run_dir}")
    g = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)
    w = g("window")

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.4), facecolor=COL_BG)
    panels = [
        (axs[0][0], "training loss", [("loss", COL_PRED, "total"),
                                      ("acc_loss", COL_PRICE, "accuracy term")]),
        (axs[0][1], "accuracy", [("band_accuracy", COL_OK, "within 10 bps"),
                                 ("direction_acc", COL_PRICE, "direction")]),
        (axs[1][0], "forecast error (per-mille)",
         [("mae_permille", COL_WARN, "MAE"), ("rmse_permille", COL_PRED, "RMSE")]),
        (axs[1][1], "skill vs naive random walk",
         [("skill_vs_naive", COL_OK, "skill")]),
    ]
    for ax, title, keys in panels:
        _style(ax, title)
        smoothed = []
        for k, c, lbl in keys:
            y = g(k)
            if not np.isfinite(y).any():
                continue
            ax.plot(w, y, color=c, lw=0.6, alpha=0.22)
            s = _ema(y, smooth)
            smoothed.append(s)
            ax.plot(w, s, color=c, lw=1.8, label=f"{lbl} (EMA{smooth})")
        if smoothed:
            _robust_ylim(ax, [g(k) for k, _, _ in keys])
        if title.startswith("accuracy"):
            ax.axhline(0.5, color=COL_FG, lw=0.7, ls="--", alpha=0.45)
            ax.set_ylim(0, 1)
        if title.startswith("skill"):
            ax.axhline(0.0, color=COL_FG, lw=0.8, ls="--", alpha=0.6)
            ax.set_ylim(-0.6, 0.6)
            ax2 = ax.twinx()
            ax2.plot(w, _ema(g("jitter_permille"), smooth), color=COL_PRICE,
                     lw=1.3, label="jitter (right)")
            ax2.tick_params(colors=COL_PRICE, labelsize=7)
            ax2.set_ylabel("jitter per-mille/s", color=COL_PRICE, fontsize=7)
            for s in ax2.spines.values():
                s.set_color(COL_GRID)
        if ax.get_legend_handles_labels()[0]:
            leg = ax.legend(fontsize=7, facecolor=COL_BG, edgecolor=COL_GRID,
                            loc="upper left")
            for t in leg.get_texts():
                t.set_color(COL_FG)
        else:
            ax.text(0.5, 0.5, "not recorded for this run", color=COL_GRID,
                    fontsize=9, ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xlabel("window (30 simulated minutes each)", color=COL_FG,
                      fontsize=8)
    fig.suptitle(f"BTCUSDT 30-minute-ahead predictor  --  "
                 f"{os.path.basename(os.path.abspath(run_dir))}",
                 color=COL_FG, fontsize=11, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=COL_BG)
    plt.close(fig)
    return out_path


# ==========================================================================
class LiveChart:
    """
    Live-updating chart for a notebook (Snowflake included) or a script.

    >>> chart = LiveChart(horizon=1800)
    >>> chart.update(ts, price, pred, sigma, metrics)     # call every window
    """

    def __init__(self, window_s: int = 5400, figsize=(11, 6.2),
                 notebook: Optional[bool] = None, horizon: int = HORIZON):
        import matplotlib
        self.notebook = (notebook if notebook is not None
                         else _in_notebook())
        if not self.notebook:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.window_s = window_s
        self.horizon = horizon
        self.fig = plt.figure(figsize=figsize, facecolor=COL_BG)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2.3, 1.0],
                                   width_ratios=[2.2, 1.0], hspace=0.35,
                                   wspace=0.22, left=0.07, right=0.98,
                                   top=0.93, bottom=0.09)
        self.ax_p = self.fig.add_subplot(gs[0, :])
        self.ax_l = self.fig.add_subplot(gs[1, 0])
        self.ax_b = self.fig.add_subplot(gs[1, 1])
        self.buf: Dict[str, List[float]] = {k: [] for k in
                                            ("ts", "price", "pred", "sigma",
                                             "target")}

    def update(self, ts, price, pred, sigma=None, metrics=None,
               target=None, info: str = "", max_points: int = 20000) -> None:
        self.buf["ts"] += list(np.asarray(ts).reshape(-1))
        self.buf["price"] += list(np.asarray(price).reshape(-1))
        self.buf["pred"] += list(np.asarray(pred).reshape(-1))
        n = len(np.asarray(ts).reshape(-1))
        self.buf["sigma"] += (list(np.asarray(sigma).reshape(-1))
                              if sigma is not None else [0.0] * n)
        self.buf["target"] += (list(np.asarray(target).reshape(-1))
                               if target is not None else [np.nan] * n)
        for k in self.buf:
            if len(self.buf[k]) > max_points:
                self.buf[k] = self.buf[k][-max_points:]

        s = RunSeries(ts=np.asarray(self.buf["ts"], np.int64),
                      price=np.asarray(self.buf["price"], np.float64),
                      pred_price=np.asarray(self.buf["pred"], np.float64),
                      sigma_price=np.asarray(self.buf["sigma"], np.float64),
                      target_price=np.asarray(self.buf["target"], np.float64),
                      window=np.zeros(len(self.buf["ts"]), np.int64))
        plot_frame(self.fig, (self.ax_p, self.ax_b, self.ax_l), s,
                   len(s) - 1, window_s=self.window_s,
                   metrics=metrics or {}, info=info)
        self._flush()

    def _flush(self) -> None:
        if self.notebook:
            from IPython.display import clear_output, display
            clear_output(wait=True)
            display(self.fig)
        else:
            self.fig.canvas.draw_idle()

    def save(self, path: str, dpi: int = 120) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fig.savefig(path, dpi=dpi, facecolor=COL_BG)
        return path

    def close(self) -> None:
        self.plt.close(self.fig)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ != "TerminalInteractiveShell"
    except Exception:
        return False


# ==========================================================================
def build_standalone_html(run_dir: str,
                          out_path: str = "artifacts/live_dashboard.html",
                          lane: int = 0, max_points: int = 6000,
                          title: str = "Bitcoin Price Predictor") -> str:
    """
    Write a single self-contained HTML file that *replays* the run as an
    animation.  All data is embedded; the chart is drawn with plain canvas
    2D calls.  No CDN, no fetch, no network -- so it works offline, inside
    Snowflake, and in sandboxed preview panes.
    """
    series = load_run_series(run_dir, lane=lane)
    rows = load_metrics(run_dir)
    n = len(series)
    step = max(1, n // max_points)
    sl = slice(0, n, step)

    def clean(a):
        x = np.asarray(a[sl], np.float64)
        return [None if not np.isfinite(v) else round(float(v), 2) for v in x]

    payload = {
        "ts": [int(v) for v in series.ts[sl]],
        "price": clean(series.price),
        "pred": clean(series.pred_price),
        "sigma": clean(series.sigma_price),
        "target": clean(series.target_price),
        "window": [int(v) for v in series.window[sl]],
        "horizon": HORIZON,
        "metrics": [{k: (None if isinstance(v, float) and not np.isfinite(v)
                         else v)
                     for k, v in r.items()
                     if k in ("window", "band_accuracy", "direction_acc",
                              "skill_vs_naive", "calibration", "loss",
                              "mae_permille", "jitter_permille",
                              "sim_sec_per_sec", "utc")}
                    for r in rows],
    }
    summary = {}
    sp = os.path.join(run_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            summary = json.load(fh)

    html = _HTML_TEMPLATE.replace("__TITLE__", title) \
                         .replace("__DATA__", json.dumps(payload)) \
                         .replace("__SUMMARY__", json.dumps(summary, default=str))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
 :root{--bg:#0e1117;--fg:#e6edf3;--grid:#262c36;--price:#4cc9f0;--pred:#ffb703;
       --ok:#3fb950;--warn:#f0883e;--dim:#8b949e}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.45 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:1180px;margin:0 auto;padding:18px}
 h1{font-size:19px;margin:0 0 2px}
 .sub{color:var(--dim);font-size:12px;margin-bottom:14px}
 .card{background:#11161d;border:1px solid var(--grid);border-radius:10px;
       padding:12px;margin-bottom:12px}
 canvas{width:100%;display:block;border-radius:6px}
 .row{display:flex;gap:12px;flex-wrap:wrap}
 .col{flex:1 1 320px}
 .bar{margin:9px 0}
 .bar .lab{display:flex;justify-content:space-between;font-size:12px;
           color:var(--dim);margin-bottom:3px}
 .bar .trk{height:9px;background:var(--grid);border-radius:5px;overflow:hidden}
 .bar .fil{height:100%;width:0;border-radius:5px;transition:width .18s linear}
 .ctl{display:flex;gap:10px;align-items:center;margin-top:8px;flex-wrap:wrap}
 button{background:#1f6feb;border:0;color:#fff;padding:6px 14px;border-radius:6px;
        cursor:pointer;font-size:13px}
 button.sec{background:var(--grid);color:var(--fg)}
 input[type=range]{flex:1;min-width:160px}
 .kv{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--dim)}
 .kv b{color:var(--fg);font-weight:600}
 .leg{display:flex;gap:16px;font-size:12px;color:var(--dim);margin-bottom:6px}
 .sw{display:inline-block;width:18px;height:3px;vertical-align:middle;
     margin-right:5px}
</style></head><body><div class="wrap">
<h1>__TITLE__</h1>
<div class="sub">Second-by-second replay &middot; model forecasts 30 minutes
 ahead &middot; dotted line is plotted at the time it predicts</div>

<div class="card">
 <div class="leg">
  <span><i class="sw" style="background:var(--price)"></i>actual price</span>
  <span><i class="sw" style="background:var(--pred);height:0;
        border-top:3px dotted var(--pred)"></i>predicted (t+30m)</span>
  <span><i class="sw" style="background:#ffb70355"></i>&plusmn;1&sigma;</span>
 </div>
 <canvas id="chart" height="360"></canvas>
 <div class="ctl">
  <button id="play">Pause</button>
  <button id="rst" class="sec">Restart</button>
  <input id="seek" type="range" min="0" max="100" value="0">
  <span id="clock" class="kv"></span>
 </div>
</div>

<div class="row">
 <div class="card col">
  <div style="font-size:12px;color:var(--dim);margin-bottom:6px">
   TRAINING ACCURACY</div>
  <div id="bars"></div>
 </div>
 <div class="card col">
  <div style="font-size:12px;color:var(--dim);margin-bottom:6px">RUN</div>
  <div id="stats" class="kv"></div>
  <canvas id="err" height="120" style="margin-top:10px"></canvas>
 </div>
</div>
</div>
<script>
const D = __DATA__, S = __SUMMARY__;
const BARS=[["band_accuracy","band 10bps",0,1,p=>(100*p).toFixed(1)+"%"],
            ["direction_acc","direction",.3,.8,p=>(100*p).toFixed(1)+"%"],
            ["skill_vs_naive","skill vs naive",-.5,.5,p=>p.toFixed(3)],
            ["calibration","calibration",0,1,p=>(100*p).toFixed(1)+"%"]];
const cv=document.getElementById("chart"),cx=cv.getContext("2d");
const ev=document.getElementById("err"),ex=ev.getContext("2d");
const N=D.ts.length, WIN=Math.max(60,Math.min(2400,Math.floor(N/6)));
let i=Math.min(WIN,N-1), playing=true;

function fit(c){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;
 c.width=r.width*d;c.height=c.height*d/(c._s||1);c._s=d;
 c.getContext("2d").setTransform(d,0,0,d,0,0);return[r.width,c.height/d];}

function bars(m){const h=document.getElementById("bars");
 if(!h.dataset.init){h.innerHTML=BARS.map(b=>
  `<div class="bar"><div class="lab"><span>${b[1]}</span>
   <span id="v_${b[0]}">--</span></div><div class="trk">
   <div class="fil" id="f_${b[0]}"></div></div></div>`).join("");
  h.dataset.init=1;}
 BARS.forEach(([k,l,lo,hi,f])=>{let v=m&&m[k]!=null?m[k]:lo;
  let p=Math.max(0,Math.min(1,(v-lo)/(hi-lo)));
  const el=document.getElementById("f_"+k);
  el.style.width=(100*p)+"%";
  el.style.background=p>.5?"var(--ok)":"var(--warn)";
  document.getElementById("v_"+k).textContent=f(v);});}

function metricAt(w){let best=null,bd=1e18;
 for(const m of D.metrics){const d=Math.abs((m.window||0)-w);
  if(d<bd){bd=d;best=m;}}return best||{};}

function draw(){
 const [W,H]=fit(cv);
 cx.clearRect(0,0,W,H);
 const a=Math.max(0,i-WIN), xs=[], hor=D.horizon;
 let lo=1e18,hi=-1e18;
 for(let k=a;k<=i;k++){const p=D.price[k],q=D.pred[k],s=D.sigma[k]||0;
  if(p!=null){lo=Math.min(lo,p);hi=Math.max(hi,p);}
  if(q!=null){lo=Math.min(lo,q-s);hi=Math.max(hi,q+s);}}
 if(!isFinite(lo)||!isFinite(hi)){return;}
 const pad=(hi-lo)*.12+1e-6; lo-=pad; hi+=pad;
 const t0=D.ts[a], t1=D.ts[i]+hor;
 const PX=52, PY=12, PB=24;
 const X=t=>PX+(t-t0)/(t1-t0)*(W-PX-10);
 const Y=v=>PY+(hi-v)/(hi-lo)*(H-PY-PB);
 cx.strokeStyle="#262c36";cx.lineWidth=1;cx.fillStyle="#8b949e";
 cx.font="10px ui-sans-serif";
 for(let g=0;g<=4;g++){const v=lo+(hi-lo)*g/4,y=Y(v);
  cx.beginPath();cx.moveTo(PX,y);cx.lineTo(W-10,y);cx.stroke();
  cx.fillText(v.toFixed(0),4,y+3);}
 // sigma ribbon
 cx.beginPath();let st=false;
 for(let k=a;k<=i;k++){const q=D.pred[k];if(q==null)continue;
  const s=D.sigma[k]||0,x=X(D.ts[k]+hor);
  if(!st){cx.moveTo(x,Y(q+s));st=true;}else cx.lineTo(x,Y(q+s));}
 for(let k=i;k>=a;k--){const q=D.pred[k];if(q==null)continue;
  const s=D.sigma[k]||0;cx.lineTo(X(D.ts[k]+hor),Y(q-s));}
 cx.closePath();cx.fillStyle="rgba(255,183,3,.17)";cx.fill();
 // actual
 cx.beginPath();cx.strokeStyle="#4cc9f0";cx.lineWidth=1.6;st=false;
 for(let k=a;k<=i;k++){const p=D.price[k];if(p==null)continue;
  const x=X(D.ts[k]),y=Y(p);st?cx.lineTo(x,y):(cx.moveTo(x,y),st=true);}
 cx.stroke();
 // predicted, dotted
 cx.beginPath();cx.strokeStyle="#ffb703";cx.lineWidth=1.7;
 cx.setLineDash([3,3]);st=false;
 for(let k=a;k<=i;k++){const q=D.pred[k];if(q==null)continue;
  const x=X(D.ts[k]+hor),y=Y(q);st?cx.lineTo(x,y):(cx.moveTo(x,y),st=true);}
 cx.stroke();cx.setLineDash([]);
 // now marker
 const xn=X(D.ts[i]);cx.strokeStyle="rgba(230,237,243,.45)";cx.lineWidth=1;
 cx.beginPath();cx.moveTo(xn,PY);cx.lineTo(xn,H-PB);cx.stroke();
 cx.fillStyle="#e6edf3";cx.fillText("now",xn+3,PY+10);
 // error strip
 const [EW,EH]=fit(ev);ex.clearRect(0,0,EW,EH);
 let em=1e-9;const errs=[];
 for(let k=a;k<=i;k++){const q=D.pred[k],t=D.target[k];
  if(q==null||t==null){errs.push(null);continue;}
  const e=1000*Math.log(q/t);errs.push(e);em=Math.max(em,Math.abs(e));}
 ex.strokeStyle="#262c36";ex.beginPath();ex.moveTo(0,EH/2);
 ex.lineTo(EW,EH/2);ex.stroke();
 ex.beginPath();ex.strokeStyle="#f0883e";ex.lineWidth=1.2;st=false;
 errs.forEach((e,k)=>{if(e==null){st=false;return;}
  const x=k/Math.max(1,errs.length-1)*EW, y=EH/2-e/em*(EH/2-6);
  st?ex.lineTo(x,y):(ex.moveTo(x,y),st=true);});
 ex.stroke();
 ex.fillStyle="#8b949e";ex.font="10px ui-sans-serif";
 ex.fillText("forecast error +/-"+em.toFixed(2)+" per-mille",4,11);

 const m=metricAt(D.window[i]);bars(m);
 const dt=new Date(D.ts[i]*1000).toISOString().replace("T"," ").slice(0,19);
 document.getElementById("clock").innerHTML=
  `<b>${dt} UTC</b> &middot; window ${D.window[i]} &middot; ${i+1}/${N} s`;
 document.getElementById("stats").innerHTML=
  `<span>backend <b>${S.backend||"-"}</b></span>
   <span>params <b>${(S.params||0).toLocaleString()}</b></span>
   <span>tier <b>${S.tier||"-"}</b></span>
   <span>windows <b>${S.windows||0}</b></span>
   <span>sim days <b>${(S.simulated_days||0).toFixed(2)}</b></span>
   <span>throughput <b>${(m.sim_sec_per_sec||0).toLocaleString()} sim-s/s</b></span>
   <span>loss <b>${m.loss!=null?(+m.loss).toFixed(4):"-"}</b></span>`;
 document.getElementById("seek").value=String(Math.round(100*i/(N-1)));
}
function tick(){if(playing){i+=Math.max(1,Math.floor(N/900));
 if(i>=N){i=Math.min(WIN,N-1);} draw();} }
document.getElementById("play").onclick=e=>{playing=!playing;
 e.target.textContent=playing?"Pause":"Play";};
document.getElementById("rst").onclick=()=>{i=Math.min(WIN,N-1);draw();};
document.getElementById("seek").oninput=e=>{playing=false;
 document.getElementById("play").textContent="Play";
 i=Math.max(1,Math.round(e.target.value/100*(N-1)));draw();};
window.addEventListener("resize",draw);
draw();setInterval(tick,70);
</script></body></html>
"""
