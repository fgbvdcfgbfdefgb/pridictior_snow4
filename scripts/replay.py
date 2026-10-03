#!/usr/bin/env python3
"""
replay.py -- run a trained checkpoint over any date range, live.

Training walks the tape from 2017 forward; what you usually want to *look at*
is the trained model working on recent data it has never seen. This script
does exactly that: load a checkpoint, replay a window of market, forecast every
second, score it against what actually happened, and leave behind a normal run
store that `btcpred.viz` can animate.

No learning happens here -- weights are frozen -- so this is also the honest
way to evaluate.

Examples
--------
    # last 7 days of the dataset, using the newest checkpoint of a run
    python scripts/replay.py --run runs/demo --days 7 --render

    # a specific window, at 300x real time, with the live dashboard watching
    python scripts/replay.py --run runs/demo \
        --from 2026-09-01 --to 2026-09-03 --speed 300
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from btcpred.analyser import MarketAnalyser, N_FEATURES, RunningNormaliser  # noqa: E402
from btcpred.compact import ChunkIndex                                      # noqa: E402
from btcpred.env import autoscale, detect                                   # noqa: E402
from btcpred.metrics import RunningMetrics, evaluate_window, render_bars    # noqa: E402
from btcpred.model import build_predictor, price_to_returns, returns_to_price  # noqa: E402
from btcpred.simulator import MarketSimulator                               # noqa: E402
from btcpred.stability import StabilityConfig, StabilityFilter              # noqa: E402
from btcpred.storage import RunStore                                        # noqa: E402


def parse_day(s: str) -> int:
    return int(dt.datetime.strptime(s, "%Y-%m-%d")
               .replace(tzinfo=dt.timezone.utc).timestamp())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory with checkpoints/")
    ap.add_argument("--checkpoint", default=None, help="default: newest")
    ap.add_argument("--data-dir", default="data/btcusdt_1s")
    ap.add_argument("--out-dir", default=None, help="default: <run>/../")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--from", dest="t_from", default=None, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="t_to", default=None, help="YYYY-MM-DD")
    ap.add_argument("--days", type=float, default=3.0,
                    help="if --from/--to omitted, replay the last N days")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="simulated seconds per real second (0 = max)")
    ap.add_argument("--warmup", type=int, default=7200)
    ap.add_argument("--band-bps", type=float, default=10.0)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    index = ChunkIndex(args.data_dir, cache_chunks=3)
    print(f"[replay] dataset: {index.describe()}")

    if args.t_from and args.t_to:
        t0, t1 = parse_day(args.t_from), parse_day(args.t_to)
    else:
        t1 = index.t_end
        t0 = t1 - int(args.days * 86400)
    t0 = max(t0, index.t_start + args.warmup)
    t1 = min(t1, index.t_end)
    t1 -= (t1 - t0) % 1800
    f = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    print(f"[replay] window: {f(t0)} -> {f(t1)} UTC "
          f"({(t1-t0)/86400:.2f} days, {(t1-t0)//1800} epochs)")

    # ---- restore the trained state ---------------------------------------
    src = RunStore(os.path.dirname(os.path.abspath(args.run)),
                   run_id=os.path.basename(os.path.abspath(args.run)),
                   resume=True)
    blob = src.load_checkpoint(args.checkpoint)
    if not blob:
        print(f"!! no checkpoint under {args.run}/checkpoints")
        return 2
    hw = detect()
    auto_cfg = blob.get("autoconfig") or {}
    auto = autoscale(hw, n_features=N_FEATURES,
                     max_tier=auto_cfg.get("tier"))
    for k in ("d_model", "n_layers", "n_heads", "tcn_channels"):
        if k in auto_cfg:
            setattr(auto, k, int(auto_cfg[k]))
    model = build_predictor(N_FEATURES, auto, device="cpu" if not hw.has_cuda
                            else auto.device, backend="auto")
    model.load_state_dict(blob["model"])
    norm = RunningNormaliser(N_FEATURES)
    norm.load_state_dict(blob["normaliser"])
    print(f"[replay] restored window {int(np.asarray(blob.get('window', 0)))} "
          f"| backend={model.backend} | params={model.n_params():,}")

    # ---- replay -----------------------------------------------------------
    out_root = args.out_dir or os.path.dirname(os.path.abspath(args.run))
    store = RunStore(out_root, run_id=args.run_id or time.strftime("replay_%Y%m%d_%H%M%S"))
    sim = MarketSimulator(index, t_from=t0, t_to=t1, speed=args.speed,
                          warmup=args.warmup)
    an = MarketAnalyser()
    hist = sim.history_before(t0, args.warmup)
    if len(hist):
        an.warmup(hist)
    flt = StabilityFilter(StabilityConfig())
    if blob.get("filter"):
        try:
            flt.load_state_dict(blob["filter"])
        except Exception:
            pass
    running = RunningMetrics(span=20)
    store.write_run_json({"kind": "replay", "source_run": os.path.abspath(args.run),
                          "span": [t0, t1], "backend": model.backend,
                          "params": model.n_params(), "tier": auto.tier,
                          "dataset": index.describe()})

    carry, prev, n_win = None, None, 0
    t_begin = time.time()
    for w in sim.windows():
        feats = an.update_block(w.bars)
        x = norm.apply(feats)
        mu, sg, carry = model.predict(x, carry)
        mu = np.asarray(mu, np.float64).reshape(-1)
        sg = np.asarray(sg, np.float64).reshape(-1)
        mu_s, _ = flt.apply_block(mu, sg)
        close = np.asarray(w.bars.close, np.float64)
        pred_price = returns_to_price(close, mu_s)

        # labels for this window mature 1800 s later -- fetch them directly
        tgt = None
        if w.t_end + 1800 <= index.t_end:
            tgt = np.asarray(index.range(w.t0 + 1800, w.t_end + 1800).close,
                             np.float64)
        y = price_to_returns(close, tgt) if tgt is not None else None
        store.save_window(w.index * 1000, ts=w.bars.timestamps(), price_now=close,
                          mu_raw=mu, mu_smooth=mu_s, sigma=sg,
                          pred_price=pred_price,
                          true_future_price=tgt, target_permille=y)
        if y is not None:
            m = evaluate_window(mu_s, y, close, sg, band_bps=args.band_bps)
            running.update(m)
            row = {"window": n_win, "wall_time": round(time.time() - t_begin, 2),
                   "t0": w.t0, "t_end": w.t_end,
                   "utc": f(w.t0),
                   "sim_sec_per_sec": round(len(w.bars) /
                                            max(1e-9, time.time() - t_begin) * (n_win + 1), 1),
                   **{k: round(v, 6) for k, v in m.to_dict().items() if k != "n"}}
            store.append_metrics(row)
            store.write_live_state({
                "window": n_win, "utc": f(w.t0),
                "ts": w.bars.timestamps()[::8].tolist(),
                "price": close[::8].round(2).tolist(),
                "pred": pred_price[::8].round(2).tolist(),
                "metrics": dict(running.values),
                "throughput": row["sim_sec_per_sec"], "loss": None,
                "elapsed": row["wall_time"]})
        n_win += 1
        if n_win % 20 == 0:
            v = running.values
            print(f"  [{n_win:>5}] {f(w.t0)}  band={v.get('band_accuracy',0):.1%}  "
                  f"dir={v.get('direction_acc',0):.1%}  "
                  f"skill={v.get('skill_vs_naive',0):+.3f}  "
                  f"mae={v.get('mae_permille',0):.3f}permille")

    print(f"\n[replay] {n_win} windows in {time.time()-t_begin:.0f}s")
    print(render_bars(running.values))
    summary = {"run_id": store.run_id, "kind": "replay", "windows": n_win,
               "simulated_days": round(n_win * 1800 / 86400, 3),
               "backend": model.backend, "params": model.n_params(),
               "tier": auto.tier, "span_utc": [f(t0), f(t1)],
               "metrics": running.snapshot()}
    with open(os.path.join(store.root, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print("[replay] stored at", store.root)

    if args.render:
        from btcpred.viz import animate_run, build_standalone_html, plot_summary
        art = "artifacts"
        print("[replay] curves ->", plot_summary(store.root, f"{art}/replay_curves.png"))
        print("[replay] html   ->", build_standalone_html(
            store.root, f"{art}/live_dashboard.html",
            title="Bitcoin Price Predictor -- trained model, held-out replay"))
        print("[replay] anim   ->", animate_run(store.root,
                                                f"{art}/live_prediction.gif",
                                                frames=120, fps=12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
