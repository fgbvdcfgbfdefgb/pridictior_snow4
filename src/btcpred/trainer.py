"""
trainer.py -- the training loop.

The contract, restated precisely
--------------------------------
* **epoch = 30 minutes of simulated market time** (1800 one-second bars);
* the model emits a **fresh forecast every single second** as the simulated
  feed arrives, predicting the price **1800 s ahead**;
* those 1800 per-second forecasts are **all stored**;
* the ground truth for a forecast made at second *t* only exists at *t+1800* --
  i.e. somewhere inside the *next* window -- so window *k* is scored, turned
  into a reward/loss signal and back-propagated at the end of window *k+1*.
  That delayed-label structure is exactly what "every per-second prediction
  within that window is stored and used to compute the reward/loss signal for
  the next training cycle" describes;
* a **checkpoint is written every window**.

How it stays fast
-----------------
Because the network is strictly causal, evaluating all 1800 seconds of a window
in one batched forward is numerically identical to stepping it one second at a
time -- the *semantics* are per-second, the *execution* is vectorised.  On top
of that the rank runs several **lanes**: independent positions in the tape
(different years, different regimes) advanced in lock-step as one batch, which
is what actually fills a GPU.  ``world_size`` ranks then all-reduce gradients
through DDP, several of them sharing the same physical card.

Mode summary::

    rank 0 ─┬─ lane 0  (2017-08 →)  analyser ─┐
            ├─ lane 1  (2019-03 →)  analyser ─┼─► batched (B,1800,F) ─► GPU
            └─ lane 2  (2021-11 →)  analyser ─┘         model          │
    rank 1 ─ … (same, different shard) ─────────────────────────── all-reduce
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .analyser import MarketAnalyser, N_FEATURES, RunningNormaliser
from .compact import ChunkIndex
from .distributed import DistContext, describe_plan, launch
from .env import AutoConfig, HardwareProfile, autoscale, detect
from .metrics import (RunningMetrics, WindowMetrics, evaluate_window,
                      render_bars)
from .model import (HORIZON, LossConfig, build_predictor, price_to_returns,
                    returns_to_price)
from .simulator import EPOCH_SECONDS, MarketSimulator, split_train_eval
from .stability import StabilityConfig, StabilityFilter
from .storage import RunStore

# A decoded bar costs 7 float32 + 1 int32 + 1 bool = 33 bytes; round up for
# the transient copies made while slicing a window out of a chunk.
DECODED_BYTES_PER_BAR = 40
CHUNK_CACHE_RAM_FRACTION = 0.22


# ==========================================================================
@dataclass
class TrainConfig:
    data_dir: str = "data/btcusdt_1s"
    run_dir: str = "runs"
    run_id: Optional[str] = None

    epoch_seconds: int = EPOCH_SECONDS       # 30 minutes
    horizon_seconds: int = HORIZON           # predict 30 minutes ahead
    warmup_seconds: int = 7200               # analyser priming history

    max_windows: Optional[int] = None        # None = the entire dataset
    max_minutes: Optional[float] = None      # wall-clock budget
    eval_fraction: float = 0.05
    eval_windows: int = 48

    lr: float = 2e-3
    backend: str = "auto"                    # auto | torch | numpy
    max_tier: Optional[str] = None
    world_size: Optional[int] = None
    lanes: Optional[int] = None              # parallel tape positions per rank
    speed: float = 0.0                       # 0 = max speed; else sim-sec/sec

    checkpoint_every: int = 1                # windows (1 => every 30 min)
    keep_last_checkpoints: int = 0           # 0 = keep everything
    log_every: int = 1
    live_state: bool = True
    save_predictions: bool = True

    band_bps: float = 10.0
    seed: int = 1234
    resume: bool = False
    stability: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ==========================================================================
class _Lane:
    """One independent position in the market tape."""

    def __init__(self, lane_id: int, index: ChunkIndex, t_from: int, t_to: int,
                 cfg: TrainConfig):
        self.id = lane_id
        self.sim = MarketSimulator(index, t_from=t_from, t_to=t_to,
                                   speed=cfg.speed,
                                   epoch_seconds=cfg.epoch_seconds,
                                   warmup=cfg.warmup_seconds)
        self.analyser = MarketAnalyser()
        self.filter = StabilityFilter(StabilityConfig(**cfg.stability))
        self.gen = self.sim.windows()
        self.pending: Optional[Dict[str, Any]] = None
        self.expect_t: Optional[int] = None
        # prime the long EMAs so the first real second is already meaningful
        hist = self.sim.history_before(t_from, cfg.warmup_seconds)
        if len(hist):
            self.analyser.warmup(hist)

    def next_window(self):
        return next(self.gen, None)


# ==========================================================================
class Trainer:
    """Drives one rank."""

    def __init__(self, cfg: TrainConfig, ctx: DistContext,
                 auto: Optional[AutoConfig] = None,
                 hw: Optional[HardwareProfile] = None):
        self.cfg = cfg
        self.ctx = ctx
        self.hw = hw or detect()
        self.auto = auto or autoscale(self.hw, n_features=N_FEATURES,
                                      max_tier=cfg.max_tier,
                                      force_world_size=cfg.world_size)
        np.random.seed(cfg.seed + ctx.rank)
        try:
            import torch
            torch.manual_seed(cfg.seed + ctx.rank)
        except Exception:
            pass

        # Each lane sits in a different month, so every lane needs its own
        # decoded chunk resident. A decoded month is ~88 MB, so the number of
        # lanes is bounded by RAM, not just by compute -- without this a
        # 10-lane run thrashes the LRU and OOMs a small container.
        probe = ChunkIndex(cfg.data_dir, cache_chunks=1)
        bytes_per_chunk = max(1, int(np.median([c["n"] for c in probe.chunks])
                                     * DECODED_BYTES_PER_BAR))
        budget = self.hw.ram_gb * 1e9 * CHUNK_CACHE_RAM_FRACTION
        max_resident = int(max(2, min(16, budget // bytes_per_chunk)))

        self.n_lanes = int(cfg.lanes or max(1, min(self.auto.batch_streams, 8)))
        if self.n_lanes > max_resident:
            if ctx.is_main:
                print(f"[trainer] capping lanes {self.n_lanes} -> {max_resident} "
                      f"({self.hw.ram_gb:.1f} GB RAM, "
                      f"{bytes_per_chunk/1e6:.0f} MB per decoded month)")
            self.n_lanes = max_resident
        self.index = ChunkIndex(cfg.data_dir,
                                cache_chunks=max(3, self.n_lanes + 1))
        (self.t_train0, self.t_train1), (self.t_eval0, self.t_eval1) = \
            split_train_eval(self.index, cfg.eval_fraction, cfg.warmup_seconds)
        self.lanes: List[_Lane] = self._make_lanes()

        self.model = build_predictor(
            N_FEATURES, self.auto, device=ctx.device,
            backend=cfg.backend, ddp=ctx.is_distributed, lr=cfg.lr,
            loss_cfg=LossConfig())
        self.norm = RunningNormaliser(N_FEATURES)
        self.running = RunningMetrics(span=20)

        self.store = RunStore(cfg.run_dir, run_id=cfg.run_id, rank=ctx.rank,
                              resume=cfg.resume)
        self.window_counter = 0
        self.t_begin = time.time()
        self.sim_seconds = 0
        self.history: List[Dict[str, Any]] = []

        if ctx.is_main:
            self.store.write_run_json({
                "config": cfg.to_dict(),
                "hardware": asdict(self.hw),
                "autoconfig": self.auto.to_dict(),
                "features": N_FEATURES,
                "backend": self.model.backend,
                "n_params": self.model.n_params(),
                "world_size": ctx.world_size,
                "lanes_per_rank": self.n_lanes,
                "plan": describe_plan(ctx.world_size),
                "dataset": self.index.describe(),
                "train_span": [self.t_train0, self.t_train1],
                "eval_span": [self.t_eval0, self.t_eval1],
            })

    # -- lane layout --------------------------------------------------------
    def _make_lanes(self) -> List[_Lane]:
        """
        Cut the training span into ``world_size * lanes`` contiguous segments
        and hand this rank its slice.  Lanes therefore sit in different market
        eras, which keeps every gradient step regime-diverse.
        """
        total = self.t_train1 - self.t_train0
        n_seg = max(1, self.ctx.world_size * self.n_lanes)
        seg = total // n_seg
        W = self.cfg.epoch_seconds
        seg -= seg % W
        if seg < W * 4:                       # tiny dataset -> fewer lanes
            n_seg = max(1, total // (W * 4))
            self.n_lanes = max(1, n_seg // max(1, self.ctx.world_size))
            seg = max(W * 2, (total // max(1, n_seg)) - (total // max(1, n_seg)) % W)
        lanes = []
        for i in range(self.n_lanes):
            k = self.ctx.rank * self.n_lanes + i
            a = self.t_train0 + k * seg
            b = min(self.t_train1, a + seg)
            if b - a < W * 2:
                continue
            lanes.append(_Lane(i, self.index, a, b, self.cfg))
        if not lanes:                          # degenerate: one lane, everything
            lanes = [_Lane(0, self.index, self.t_train0, self.t_train1, self.cfg)]
        return lanes

    # -- ground truth -------------------------------------------------------
    def _targets_for(self, t0: int, n: int, cur_close: Optional[np.ndarray],
                     cur_t0: Optional[int]) -> Optional[np.ndarray]:
        """
        Close prices 1800 s after each second of the window starting at ``t0``.

        When the next window is temporally adjacent (the normal case) its own
        closes *are* the targets -- free.  At a lane boundary we fetch them
        from the index instead.
        """
        H = self.cfg.horizon_seconds
        if cur_close is not None and cur_t0 == t0 + H and len(cur_close) >= n:
            return np.asarray(cur_close[:n], np.float64)
        a, b = t0 + H, t0 + H + n
        if b > self.index.t_end:
            return None
        return np.asarray(self.index.range(a, b).close, np.float64)

    # -- one synchronous step across all lanes ------------------------------
    def step(self) -> Optional[Dict[str, Any]]:
        """
        Advance every lane by one 30-minute window.

        1. pull the window, run the CPU analyser, normalise;
        2. forward pass -> a forecast for every second (inference, no grad);
        3. stabilise, convert to prices, persist all 1800 predictions;
        4. score the *previous* window (its labels have just matured) and take
           one optimisation step on it;
        5. checkpoint.
        """
        cfg = self.cfg
        t_step = time.time()
        wins, feats, lanes_live = [], [], []
        for lane in self.lanes:
            w = lane.next_window()
            if w is None:
                continue
            # a discontinuity means the analyser must be re-primed
            if lane.expect_t is not None and w.t0 != lane.expect_t:
                hist = lane.sim.history_before(w.t0, cfg.warmup_seconds)
                lane.analyser = MarketAnalyser()
                if len(hist):
                    lane.analyser.warmup(hist)
                lane.pending = None
            lane.expect_t = w.t_end
            f = lane.analyser.update_block(w.bars)          # CPU analyser
            wins.append(w)
            feats.append(f)
            lanes_live.append(lane)
        if not wins:
            return None

        X = np.stack(feats, axis=0)                          # (B, T, F)
        self.norm.observe(X.reshape(-1, N_FEATURES)[::17])   # cheap subsample
        Xn = self.norm.apply(X)

        # ---- 2. forecast every second ------------------------------------
        carry = getattr(self, "_carry", None)
        mu, sigma, carry = self.model.predict(Xn, carry)
        self._carry = carry
        mu = np.atleast_2d(np.asarray(mu, np.float64))
        sigma = np.atleast_2d(np.asarray(sigma, np.float64))

        # ---- 3. stabilise + persist --------------------------------------
        learn_X, learn_Y, learn_W = [], [], []
        metrics_acc: List[WindowMetrics] = []
        for b, (lane, w) in enumerate(zip(lanes_live, wins)):
            mu_s, stab = lane.filter.apply_block(mu[b], sigma[b])
            close = np.asarray(w.bars.close, np.float64)
            pred_price = returns_to_price(close, mu_s)
            gi = w.index * 1000 + lane.id       # globally unique window id
            if cfg.save_predictions:
                self.store.save_window(
                    gi, ts=w.bars.timestamps(), price_now=close,
                    mu_raw=mu[b], mu_smooth=mu_s, sigma=sigma[b],
                    pred_price=pred_price)

            # ---- 4. the previous window's labels have now matured ---------
            prev = lane.pending
            if prev is not None:
                tgt_price = self._targets_for(prev["t0"], prev["n"],
                                              close, w.t0)
                if tgt_price is not None:
                    y = price_to_returns(prev["close"], tgt_price)
                    learn_X.append(prev["x"])
                    learn_Y.append(y)
                    learn_W.append(np.ones_like(y))
                    m = evaluate_window(prev["mu_s"], y, prev["close"],
                                        prev["sigma"], band_bps=cfg.band_bps)
                    metrics_acc.append(m)
                    if cfg.save_predictions:
                        self.store.save_window(
                            prev["gi"], ts=prev["ts"], price_now=prev["close"],
                            mu_raw=prev["mu"], mu_smooth=prev["mu_s"],
                            sigma=prev["sigma"], pred_price=prev["pred_price"],
                            true_future_price=tgt_price, target_permille=y)

            lane.pending = {
                "gi": gi, "t0": w.t0, "n": len(w.bars), "x": Xn[b],
                "close": close, "mu": mu[b], "mu_s": mu_s, "sigma": sigma[b],
                "pred_price": pred_price, "ts": w.bars.timestamps(),
                "stab": stab,
            }

        # ---- optimisation step on the matured windows ---------------------
        stats: Dict[str, float] = {}
        if learn_X:
            stats = self.model.learn(np.stack(learn_X, 0), np.stack(learn_Y, 0),
                                     carry=None, weights=np.stack(learn_W, 0))

        self.window_counter += 1
        self.sim_seconds += sum(len(w.bars) for w in wins)
        elapsed = max(1e-9, time.time() - t_step)

        agg: Dict[str, Any] = {}
        if metrics_acc:
            keys = metrics_acc[0].to_dict().keys()
            agg = {k: float(np.mean([m.to_dict()[k] for m in metrics_acc]))
                   for k in keys}
            self.running.update(agg)
        row = {
            # lanes sit in different eras, so the logged span is lane 0's
            "window": self.window_counter,
            "wall_time": round(time.time() - self.t_begin, 3),
            "t0": wins[0].t0, "t_end": wins[0].t_end,
            "utc": wins[0].utc_range()[0].strftime("%Y-%m-%d %H:%M:%S"),
            "sim_sec_per_sec": round(sum(len(w.bars) for w in wins) / elapsed, 1),
            **{k: round(v, 6) for k, v in stats.items()},
            **{k: round(v, 6) for k, v in agg.items() if k != "n"},
        }
        self.store.append_metrics(row)
        self.history.append(row)

        # ---- 5. checkpoint every window -----------------------------------
        if (self.ctx.is_main and cfg.checkpoint_every
                and self.window_counter % cfg.checkpoint_every == 0):
            self.save_checkpoint()

        if cfg.live_state and self.ctx.is_main:
            self._write_live(wins[0], mu[0], sigma[0],
                             self.lanes[0].pending["mu_s"], row)
        return row

    # -- what the live chart draws ------------------------------------------
    def latest_frame(self, lane: int = 0) -> Optional[Dict[str, Any]]:
        """Most recent window of lane ``lane``, shaped for ``viz.LiveChart``."""
        if lane >= len(self.lanes) or self.lanes[lane].pending is None:
            return None
        p = self.lanes[lane].pending
        sig_price = p["close"] * (np.exp(np.asarray(p["sigma"]) / 1000.0) - 1.0)
        return {
            "ts": p["ts"], "price": p["close"], "pred": p["pred_price"],
            "sigma": sig_price, "metrics": dict(self.running.values),
            "window": self.window_counter,
            "utc": self.history[-1]["utc"] if self.history else "",
            "throughput": self.history[-1].get("sim_sec_per_sec", 0)
            if self.history else 0,
        }

    # -- persistence --------------------------------------------------------
    def save_checkpoint(self) -> str:
        blob = {
            "window": self.window_counter,
            "model": self.model.state_dict(),
            "normaliser": self.norm.state_dict(),
            "analyser": [l.analyser.state_dict() for l in self.lanes],
            "filter": [l.filter.state_dict() for l in self.lanes],
            "metrics": self.running.snapshot(),
            "autoconfig": self.auto.to_dict(),
            "config": self.cfg.to_dict(),
            "sim_seconds": self.sim_seconds,
        }
        return self.store.save_checkpoint(self.window_counter, blob,
                                          keep_last=self.cfg.keep_last_checkpoints)

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        blob = self.store.load_checkpoint(path)
        if not blob:
            return False
        try:
            self.model.load_state_dict(blob["model"])
            self.norm.load_state_dict(blob["normaliser"])
            self.window_counter = int(blob.get("window", 0))
            return True
        except Exception as e:                           # pragma: no cover
            print(f"[trainer] checkpoint load failed: {e}")
            return False

    def _write_live(self, w, mu, sigma, mu_s, row) -> None:
        n = len(w.bars)
        k = max(1, n // 240)                      # thin for the browser
        close = np.asarray(w.bars.close, np.float64)
        pred = returns_to_price(close, mu_s)
        self.store.write_live_state({
            "window": self.window_counter,
            "utc": row["utc"],
            "ts": w.bars.timestamps()[::k].tolist(),
            "price": close[::k].round(2).tolist(),
            "pred": pred[::k].round(2).tolist(),
            "sigma": np.asarray(sigma)[::k].round(4).tolist(),
            "metrics": {k2: self.running.get(k2) for k2 in
                        ("band_accuracy", "direction_acc", "skill_vs_naive",
                         "calibration", "mae_permille", "jitter_permille")},
            "throughput": row.get("sim_sec_per_sec", 0),
            "loss": row.get("loss", None),
            "elapsed": row["wall_time"],
        })

    # -- main loop ----------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        if cfg.resume:
            self.load_checkpoint()
        if self.ctx.is_main:
            print(f"[trainer] {describe_plan(self.ctx.world_size)}")
            print(f"[trainer] backend={self.model.backend} "
                  f"params={self.model.n_params():,} tier={self.auto.tier} "
                  f"lanes/rank={self.n_lanes} features={N_FEATURES}")
            print(f"[trainer] dataset: {self.index.describe()}")
            print(f"[trainer] run dir: {self.store.root}")
        deadline = (self.t_begin + cfg.max_minutes * 60.0
                    if cfg.max_minutes else None)
        while True:
            if cfg.max_windows and self.window_counter >= cfg.max_windows:
                break
            if deadline and time.time() > deadline:
                if self.ctx.is_main:
                    print("[trainer] wall-clock budget reached")
                break
            row = self.step()
            if row is None:
                if self.ctx.is_main:
                    print("[trainer] tape exhausted")
                break
            if self.ctx.is_main and cfg.log_every and \
                    self.window_counter % cfg.log_every == 0:
                self._log(row)
        if self.ctx.is_main:
            self.save_checkpoint()
        return self.finish()

    def _log(self, row: Dict[str, Any]) -> None:
        v = self.running.values
        print(f"[w{row['window']:>6}] {row['utc']}  "
              f"loss={row.get('loss', float('nan')):.4f}  "
              f"mae={v.get('mae_permille', float('nan')):.3f}permille  "
              f"band={v.get('band_accuracy', 0):.1%}  "
              f"dir={v.get('direction_acc', 0):.1%}  "
              f"skill={v.get('skill_vs_naive', 0):+.3f}  "
              f"jit={v.get('jitter_permille', 0):.4f}  "
              f"{row['sim_sec_per_sec']:,.0f} sim-s/s")

    # -- held-out evaluation -------------------------------------------------
    def evaluate(self, max_windows: Optional[int] = None) -> Dict[str, Any]:
        """Walk-forward evaluation on the most recent, never-trained slice."""
        cfg = self.cfg
        n = max_windows or cfg.eval_windows
        sim = MarketSimulator(self.index, t_from=self.t_eval0, t_to=self.t_eval1,
                              speed=0.0, epoch_seconds=cfg.epoch_seconds,
                              warmup=cfg.warmup_seconds)
        an = MarketAnalyser()
        hist = sim.history_before(self.t_eval0, cfg.warmup_seconds)
        if len(hist):
            an.warmup(hist)
        flt = StabilityFilter(StabilityConfig(**cfg.stability))
        carry = None
        out: List[WindowMetrics] = []
        prev = None
        for w in sim.windows(max_windows=n + 1):
            f = an.update_block(w.bars)
            x = self.norm.apply(f)
            mu, sg, carry = self.model.predict(x, carry)
            mu = np.asarray(mu, np.float64).reshape(-1)
            sg = np.asarray(sg, np.float64).reshape(-1)
            mu_s, _ = flt.apply_block(mu, sg)
            close = np.asarray(w.bars.close, np.float64)
            if prev is not None:
                tgt = self._targets_for(prev["t0"], prev["n"], close, w.t0)
                if tgt is not None:
                    y = price_to_returns(prev["close"], tgt)
                    out.append(evaluate_window(prev["mu_s"], y, prev["close"],
                                               prev["sigma"], cfg.band_bps))
            prev = {"t0": w.t0, "n": len(w.bars), "close": close,
                    "mu_s": mu_s, "sigma": sg}
        if not out:
            return {}
        keys = out[0].to_dict().keys()
        res = {k: float(np.mean([m.to_dict()[k] for m in out])) for k in keys}
        res["windows"] = len(out)
        return res

    def finish(self) -> Dict[str, Any]:
        summary = {
            "run_id": self.store.run_id,
            "rank": self.ctx.rank,
            "windows": self.window_counter,
            "simulated_seconds": self.sim_seconds,
            "simulated_days": round(self.sim_seconds / 86400.0, 3),
            "wall_seconds": round(time.time() - self.t_begin, 1),
            "backend": self.model.backend,
            "params": self.model.n_params(),
            "tier": self.auto.tier,
            "metrics": self.running.snapshot(),
        }
        if self.ctx.is_main:
            ev = self.evaluate()
            summary["holdout"] = ev
            with open(os.path.join(self.store.root, "summary.json"), "w") as fh:
                json.dump(summary, fh, indent=2, default=str)
            print("\n" + "=" * 66)
            print(f"  run {self.store.run_id} finished")
            print(f"  {self.window_counter} windows | "
                  f"{summary['simulated_days']:.2f} simulated days | "
                  f"{summary['wall_seconds']:.0f}s wall")
            print("  training accuracy (EMA):")
            print(render_bars(self.running.values))
            if ev:
                print("  held-out (never trained on):")
                print(render_bars(ev))
            print("=" * 66)
        return summary


# ==========================================================================
def _entry(ctx: DistContext, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    cfg = TrainConfig(**cfg_dict)
    return Trainer(cfg, ctx).run()


def train(cfg: TrainConfig) -> Dict[str, Any]:
    """Public entry point -- spawns ranks if the hardware justifies it."""
    hw = detect()
    auto = autoscale(hw, n_features=N_FEATURES, max_tier=cfg.max_tier,
                     force_world_size=cfg.world_size)
    world = cfg.world_size or auto.world_size
    if cfg.backend == "numpy":
        world = 1
    rdzv = os.path.join(os.path.abspath(cfg.run_dir), ".rdzv")
    return launch(_entry, world_size=world, rendezvous_dir=rdzv,
                  cfg_dict=cfg.to_dict())
