#!/usr/bin/env python3
"""
train.py -- entry point for the Bitcoin price predictor.

Runs fully offline against the compact dataset in ``data/btcusdt_1s``.
The model size, the number of distributed ranks and the batch width are all
derived from the machine it finds itself on -- there is nothing to tune to move
between a laptop, a Snowflake warehouse kernel and a GPU compute pool.

Typical use
-----------
    python train.py --probe                     # what would this box do?
    python train.py --max-windows 200           # short run
    python train.py                             # the whole tape, max speed
    python train.py --speed 60 --live           # replay 60x real time
    python train.py --resume                    # continue from last checkpoint
    python train.py --render                    # animate the finished run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from btcpred.analyser import N_FEATURES                      # noqa: E402
from btcpred.distributed import describe_plan                # noqa: E402
from btcpred.env import autoscale, describe, detect          # noqa: E402
from btcpred.trainer import TrainConfig, train               # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--data-dir", default="data/btcusdt_1s")
    d.add_argument("--run-dir", default="runs")
    d.add_argument("--run-id", default=None)
    d.add_argument("--eval-fraction", type=float, default=0.05,
                   help="newest share of the tape held out for evaluation")

    t = p.add_argument_group("schedule")
    t.add_argument("--epoch-seconds", type=int, default=1800,
                   help="one epoch = this many simulated seconds (default 30 min)")
    t.add_argument("--horizon", type=int, default=1800,
                   help="forecast this many seconds ahead (default 30 min)")
    t.add_argument("--max-windows", type=int, default=None)
    t.add_argument("--max-minutes", type=float, default=None,
                   help="wall-clock budget")
    t.add_argument("--speed", type=float, default=0.0,
                   help="simulated seconds per real second (0 = max speed)")
    t.add_argument("--warmup", type=int, default=7200)

    m = p.add_argument_group("model / hardware")
    m.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    m.add_argument("--max-tier", default=None,
                   help="pico|nano|micro|small|base|large|xl|xxl")
    m.add_argument("--world-size", type=int, default=None,
                   help="distributed ranks (default: auto)")
    m.add_argument("--lanes", type=int, default=None,
                   help="parallel tape positions per rank (default: auto)")
    m.add_argument("--lr", type=float, default=2e-3)

    o = p.add_argument_group("output")
    o.add_argument("--checkpoint-every", type=int, default=1,
                   help="windows between checkpoints (1 = every 30 min)")
    o.add_argument("--keep-last", type=int, default=0,
                   help="keep only the N newest checkpoints (0 = keep all)")
    o.add_argument("--no-predictions", action="store_true",
                   help="do not store the per-second prediction tape")
    o.add_argument("--log-every", type=int, default=10)
    o.add_argument("--live", action="store_true",
                   help="write live/state.json for the dashboard")
    o.add_argument("--render", action="store_true",
                   help="render GIF + HTML + curves when the run finishes")
    o.add_argument("--resume", action="store_true")
    o.add_argument("--probe", action="store_true",
                   help="print the hardware plan and exit")
    o.add_argument("--seed", type=int, default=1234)
    return p


def main() -> int:
    args = build_parser().parse_args()

    hw = detect()
    auto = autoscale(hw, n_features=N_FEATURES, max_tier=args.max_tier,
                     force_world_size=args.world_size)
    print(describe(hw, auto))
    print(f"[plan] {describe_plan(args.world_size or auto.world_size)}")
    if args.probe:
        print(json.dumps({"hardware": hw.__dict__, "autoconfig": auto.to_dict()},
                         indent=2, default=str))
        return 0

    if not os.path.exists(os.path.join(args.data_dir, "MANIFEST.json")):
        print(f"\n!! No dataset at {args.data_dir}.\n"
              f"   On a machine with internet run:\n"
              f"     python scripts/download_binance_1s.py --all\n"
              f"   Offline, point --data-dir at the bundled dataset directory.")
        return 2

    cfg = TrainConfig(
        data_dir=args.data_dir, run_dir=args.run_dir, run_id=args.run_id,
        epoch_seconds=args.epoch_seconds, horizon_seconds=args.horizon,
        warmup_seconds=args.warmup, max_windows=args.max_windows,
        max_minutes=args.max_minutes, eval_fraction=args.eval_fraction,
        lr=args.lr, backend=args.backend, max_tier=args.max_tier,
        world_size=args.world_size, lanes=args.lanes, speed=args.speed,
        checkpoint_every=args.checkpoint_every,
        keep_last_checkpoints=args.keep_last, log_every=args.log_every,
        live_state=args.live or True,
        save_predictions=not args.no_predictions,
        seed=args.seed, resume=args.resume,
    )
    summary = train(cfg)

    if args.render and summary:
        from btcpred.viz import animate_run, build_standalone_html, plot_summary
        run_dir = os.path.join(args.run_dir, summary["run_id"])
        try:
            print("[render] curves  ->", plot_summary(run_dir, "artifacts/training_curves.png"))
            print("[render] html    ->", build_standalone_html(run_dir, "artifacts/live_dashboard.html"))
            print("[render] anim    ->", animate_run(run_dir, "artifacts/live_prediction.gif"))
        except Exception as e:                        # noqa: BLE001
            print(f"[render] failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
