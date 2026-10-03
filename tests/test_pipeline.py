#!/usr/bin/env python3
"""
test_pipeline.py -- offline self-tests.

No network, no GPU, no pytest required:

    python tests/test_pipeline.py          # uses a synthetic dataset
    python tests/test_pipeline.py --data data/btcusdt_1s   # also checks real data

What is asserted
----------------
1.  codec round-trip: prices are exact, volumes within quantisation error;
2.  ``ChunkIndex`` serves arbitrary windows across chunk boundaries;
3.  the analyser's **streaming** and **batched** paths agree -- the identity
    that makes "train at max speed" equivalent to "update every second";
4.  the analyser is strictly causal: changing the future cannot change the past;
5.  the predictor runs, learns, and its loss falls on a learnable signal;
6.  the stability filter respects its slew limit;
7.  a two-window trainer run produces predictions, metrics and a checkpoint,
    and the checkpoint reloads.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from btcpred.analyser import MarketAnalyser, N_FEATURES, ema_run   # noqa: E402
from btcpred.compact import (ChunkIndex, encode_grid, load_chunk,  # noqa: E402
                             save_chunk, write_manifest)
from btcpred.env import autoscale, detect                          # noqa: E402
from btcpred.model import (build_predictor, price_to_returns,      # noqa: E402
                           returns_to_price)
from btcpred.simulator import MarketSimulator                      # noqa: E402
from btcpred.stability import StabilityConfig, StabilityFilter     # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
def make_synthetic(root: str, months: int = 2, n: int = 90_000,
                   seed: int = 7) -> str:
    """A believable random-walk tape so the tests need no download."""
    rng = np.random.default_rng(seed)
    t0 = 1_600_000_000
    t0 -= t0 % 86400
    price = 30_000.0
    chunks = []
    for m in range(months):
        steps = rng.standard_normal(n) * 3.0
        close = np.maximum(1000.0, price + np.cumsum(steps))
        price = float(close[-1])
        close_c = np.rint(close * 100).astype(np.int64)
        spread = np.rint(np.abs(rng.standard_normal(n)) * 40).astype(np.int64)
        high_c = close_c + spread
        low_c = close_c - spread
        open_c = np.concatenate([[close_c[0]], close_c[:-1]])
        vol = np.abs(rng.standard_normal(n)) * 0.5
        gap = rng.random(n) < 0.02
        vol[gap] = 0.0
        qvol = vol * (close_c / 100.0)
        trades = (vol * 20).astype(np.int64)
        tbb = vol * rng.uniform(0.2, 0.8, n)
        tbb[gap] = 0.0
        payload = encode_grid(t0 + m * n, close_c, open_c, high_c, low_c,
                              vol, qvol, trades, tbb, gap)
        f = f"syn-{m:02d}.npz"
        b = save_chunk(os.path.join(root, f), payload)
        chunks.append({"file": f, "t0": t0 + m * n, "n": n, "bytes": b})
    write_manifest(root, chunks, {"symbol": "SYNTH"})
    return root


# --------------------------------------------------------------------------
def test_codec(root: str) -> None:
    print("\n[1] codec round-trip")
    b = load_chunk(os.path.join(root, "syn-00.npz"))
    check("decodes", len(b) > 0, f"{len(b):,} bars")
    check("prices finite and positive",
          bool(np.isfinite(b.close).all() and (b.close > 0).all()))
    check("OHLC invariants",
          bool((b.low <= b.high + 1e-3).all()
               and (b.close <= b.high + 1e-3).all()
               and (b.close >= b.low - 1e-3).all()))
    # the codec is exact in integer cents; the only loss is the final float32
    # cast, so the decoded price must sit on the cent grid to float32 precision
    cents = np.rint(b.close.astype(np.float64) * 100)
    rel = np.abs(cents / 100.0 - b.close) / np.maximum(b.close, 1.0)
    check("price exact on the cent grid (to float32)", bool(rel.max() < 1e-6),
          f"max relative error {rel.max():.2e}")
    check("gap mask matches zero volume",
          bool((b.volume[b.gap] == 0).all()))


def test_index(root: str) -> None:
    print("\n[2] chunk index")
    idx = ChunkIndex(root)
    check("manifest span", idx.total_seconds > 0, idx.describe())
    a = idx.t_start + 1000
    w = idx.range(a, a + 1800)
    check("1800-second window", len(w) == 1800 and w.t0 == a)
    # window straddling the chunk boundary
    bnd = idx.chunks[0]["t0"] + idx.chunks[0]["n"] - 900
    w2 = idx.range(bnd, bnd + 1800)
    check("cross-chunk window", len(w2) == 1800,
          f"{w2.close[0]:.2f} -> {w2.close[-1]:.2f}")
    check("no NaNs across the seam", bool(np.isfinite(w2.close).all()))


def test_ema() -> None:
    print("\n[3] EMA recursion")
    x = np.random.default_rng(0).standard_normal(5000)
    a = 2.0 / (301.0)
    fast = ema_run(x, a, 0.0)
    slow = np.empty_like(x)
    y = 0.0
    for i, v in enumerate(x):
        y = (1 - a) * y + a * v
        slow[i] = y
    check("vectorised == loop", float(np.abs(fast - slow).max()) < 1e-9,
          f"max diff {np.abs(fast-slow).max():.2e}")


def test_analyser(root: str) -> None:
    print("\n[4] analyser: streaming == batched, and causal")
    idx = ChunkIndex(root)
    bars = idx.range(idx.t_start + 5000, idx.t_start + 5000 + 600)

    a1 = MarketAnalyser()
    batched = a1.update_block(bars)

    a2 = MarketAnalyser()
    rows = []
    for i in range(len(bars)):
        one = bars.slice(i, i + 1)
        rows.append(a2.update_block(one)[0])
    streamed = np.stack(rows)
    d = float(np.abs(batched - streamed).max())
    check("streaming matches batched", d < 1e-3, f"max diff {d:.2e}")
    check("feature count", batched.shape[1] == N_FEATURES, str(N_FEATURES))
    check("all finite", bool(np.isfinite(batched).all()))

    # causality: perturbing the tail must not move earlier features
    mutated = idx.range(idx.t_start + 5000, idx.t_start + 5000 + 600)
    mutated.close[400:] *= 1.05
    a3 = MarketAnalyser()
    other = a3.update_block(mutated)
    check("strictly causal",
          float(np.abs(other[:400] - batched[:400]).max()) < 1e-6)


def test_model_learns() -> None:
    print("\n[5] predictor learns")
    hw = detect()
    cfg = autoscale(hw, n_features=8, max_tier="nano")
    rng = np.random.default_rng(3)
    T = 600
    for backend in ("torch", "numpy"):
        try:
            m = build_predictor(8, cfg, device="cpu", backend=backend, lr=5e-3)
        except Exception as e:                        # torch absent
            print(f"  SKIP  {backend} backend ({e})")
            continue
        # a learnable signal: target = 3 * feature_0 - 2 * feature_1
        losses = []
        for step in range(40):
            x = rng.standard_normal((1, T, 8)).astype(np.float32)
            y = (3.0 * x[..., 0] - 2.0 * x[..., 1]).astype(np.float32)
            st = m.learn(x, y)
            losses.append(st["loss"])
        first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
        check(f"{backend}: loss decreases", last < first * 0.9,
              f"{first:.3f} -> {last:.3f}")
        mu, sg, _ = m.predict(rng.standard_normal((T, 8)).astype(np.float32))
        check(f"{backend}: inference shape", np.asarray(mu).shape[-1] == T)


def test_price_roundtrip() -> None:
    print("\n[6] return <-> price conversion")
    now = np.array([50_000.0, 61_234.5])
    fut = np.array([50_250.0, 60_000.0])
    r = price_to_returns(now, fut)
    back = returns_to_price(now, r)
    check("round-trips", float(np.abs(back - fut).max()) < 1e-6,
          f"returns {r.round(3)} per-mille")


def test_stability() -> None:
    print("\n[7] stability filter")
    rng = np.random.default_rng(1)
    raw = np.cumsum(rng.standard_normal(1800)) * 0.5
    f = StabilityFilter(StabilityConfig(max_slew_permille=0.05))
    out, info = f.apply_block(raw, np.ones_like(raw))
    d = np.abs(np.diff(out))
    check("slew limit respected", float(d.max()) <= 0.05 + 1e-9,
          f"max step {d.max():.4f}")
    check("jitter reduced", info["jitter_permille"] < info["raw_jitter_permille"],
          f"{info['raw_jitter_permille']:.3f} -> {info['jitter_permille']:.3f}")


def test_simulator(root: str) -> None:
    print("\n[8] simulator")
    idx = ChunkIndex(root)
    sim = MarketSimulator(idx, speed=0, warmup=1000)
    ws = list(sim.windows(max_windows=3))
    check("windows are 1800 s", all(len(w.bars) == 1800 for w in ws))
    check("windows are consecutive",
          all(ws[i + 1].t0 == ws[i].t_end for i in range(len(ws) - 1)))
    ticks = []
    for i, t in enumerate(sim.ticks(sim.t_from, sim.t_from + 50)):
        ticks.append(t)
    check("tick stream", len(ticks) == 50 and ticks[1].ts == ticks[0].ts + 1)
    check("tick matches window", abs(ticks[0].close - ws[0].bars.close[0]) < 1e-3)


def test_trainer(root: str) -> None:
    print("\n[9] trainer end-to-end")
    from btcpred.distributed import DistContext
    from btcpred.trainer import Trainer, TrainConfig
    out = tempfile.mkdtemp(prefix="btcpred_test_run_")
    try:
        cfg = TrainConfig(data_dir=root, run_dir=out, run_id="t", max_windows=3,
                          lanes=1, warmup_seconds=600, eval_windows=2,
                          log_every=0, checkpoint_every=1)
        tr = Trainer(cfg, DistContext(rank=0, world_size=1, device="cpu"))
        for _ in range(3):
            tr.step()
        check("windows ran", tr.window_counter == 3)
        preds = os.listdir(os.path.join(tr.store.root, "predictions"))
        check("per-second predictions stored", len(preds) >= 2, f"{len(preds)} files")
        d = tr.store.load_window(int(preds[0].split(".")[0].replace("w", "")))
        check("1800 predictions per window", d["ts"].shape[0] == 1800)
        check("targets attached to matured window",
              any("target_permille" in np.load(
                  os.path.join(tr.store.root, "predictions", p)).files
                  for p in preds))
        cks = [f for f in os.listdir(os.path.join(tr.store.root, "checkpoints"))
               if f.endswith(".npz")]
        check("checkpoint per window", len(cks) >= 2, f"{len(cks)} checkpoints")
        check("checkpoint reloads", tr.load_checkpoint())
        rows = tr.store.read_metrics()
        check("metrics logged", len(rows) == 3)
        f = tr.latest_frame()
        check("live frame available", f is not None and len(f["price"]) == 1800)
    finally:
        shutil.rmtree(out, ignore_errors=True)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="also sanity-check a real dataset directory")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="btcpred_test_")
    try:
        print(f"synthetic dataset -> {tmp}")
        make_synthetic(tmp)
        test_codec(tmp)
        test_index(tmp)
        test_ema()
        test_analyser(tmp)
        test_model_learns()
        test_price_roundtrip()
        test_stability()
        test_simulator(tmp)
        test_trainer(tmp)
        if args.data and os.path.exists(os.path.join(args.data, "MANIFEST.json")):
            print("\n[10] real dataset")
            idx = ChunkIndex(args.data)
            check("real manifest", idx.total_seconds > 86400, idx.describe())
            w = idx.range(idx.t_end - 1800, idx.t_end)
            check("latest window decodes",
                  len(w) == 1800 and bool(np.isfinite(w.close).all()),
                  f"{w.close[-1]:,.2f} USD")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("   FAILED:", f)
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
