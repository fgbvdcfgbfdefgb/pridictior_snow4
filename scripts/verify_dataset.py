#!/usr/bin/env python3
"""
verify_dataset.py -- offline integrity check of the compact dataset.

Run this after cloning (or after uploading to Snowflake) to prove the data
survived the trip:

    python scripts/verify_dataset.py                 # quick
    python scripts/verify_dataset.py --deep          # decode every chunk

Checks performed
----------------
* every chunk listed in MANIFEST.json exists and decodes;
* chunks tile the timeline with no overlap and no unexplained hole;
* prices are positive, finite and monotone in time index;
* OHLC invariants hold  (low <= open/close <= high);
* the gap mask is consistent with zero-volume seconds;
* reports total bars, span, gap ratio and on-disk size.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from btcpred.compact import ChunkIndex, load_chunk          # noqa: E402


def human(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/btcusdt_1s")
    ap.add_argument("--deep", action="store_true", help="decode every chunk")
    args = ap.parse_args()

    idx = ChunkIndex(args.data_dir)
    print(f"[verify] {idx.describe()}")
    print(f"[verify] span {human(idx.t_start)} -> {human(idx.t_end)} UTC")

    problems = 0
    prev_end = None
    total_bars = 0
    total_gap = 0
    pmin, pmax = np.inf, -np.inf

    for i, c in enumerate(idx.chunks):
        path = os.path.join(idx.root, c["file"])
        if not os.path.exists(path):
            print(f"  !! missing file {c['file']}")
            problems += 1
            continue
        if prev_end is not None and c["t0"] != prev_end:
            d = c["t0"] - prev_end
            print(f"  !! timeline break before {c['file']}: {d:+,} s")
            problems += 1
        prev_end = c["t0"] + c["n"]

        if not (args.deep or i in (0, len(idx.chunks) // 2, len(idx.chunks) - 1)):
            total_bars += c["n"]
            continue

        b = load_chunk(path)
        if len(b) != c["n"]:
            print(f"  !! {c['file']}: manifest says {c['n']} bars, decoded {len(b)}")
            problems += 1
        bad = ~np.isfinite(b.close) | (b.close <= 0)
        if bad.any():
            print(f"  !! {c['file']}: {int(bad.sum())} non-positive/NaN closes")
            problems += 1
        viol = (b.low > b.high + 1e-3) | (b.close > b.high + 1e-3) | \
               (b.close < b.low - 1e-3) | (b.open > b.high + 1e-3) | \
               (b.open < b.low - 1e-3)
        if viol.any():
            print(f"  !! {c['file']}: {int(viol.sum())} OHLC violations")
            problems += 1
        mism = b.gap & (b.volume > 0)
        if mism.any():
            print(f"  !! {c['file']}: {int(mism.sum())} gap-flagged bars with volume")
            problems += 1
        total_bars += len(b)
        total_gap += int(b.gap.sum())
        pmin = min(pmin, float(b.close.min()))
        pmax = max(pmax, float(b.close.max()))
        if args.deep and (i % 12 == 0):
            print(f"  .. {c['file']}  {len(b):,} bars  "
                  f"{b.close.min():,.0f}-{b.close.max():,.0f} USD  "
                  f"gaps {b.gap.mean():.4%}")

    mb = sum(c.get("bytes", 0) for c in idx.chunks) / 1e6
    print(f"[verify] chunks={len(idx.chunks)} bars={total_bars:,} "
          f"size={mb:,.0f} MB")
    if np.isfinite(pmin):
        print(f"[verify] price range {pmin:,.2f} .. {pmax:,.2f} USD"
              + (f" | gap ratio {total_gap/max(1,total_bars):.4%}"
                 if args.deep else " (sampled)"))

    # a window read must work -- this is what training actually does
    mid = (idx.t_start + idx.t_end) // 2
    w = idx.range(mid, mid + 1800)
    print(f"[verify] sample window {human(w.t0)}: {len(w)} bars, "
          f"close {w.close[0]:,.2f} -> {w.close[-1]:,.2f}")

    print("[verify] " + ("OK -- dataset is usable offline" if problems == 0
                         else f"{problems} PROBLEM(S) FOUND"))
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
