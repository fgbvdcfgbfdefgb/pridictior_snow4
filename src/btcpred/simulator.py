"""
simulator.py -- a live market simulator that replays the historical
second-by-second tape as if it were arriving right now.

Two interfaces over the exact same data, deliberately kept numerically
identical:

``MarketSimulator.ticks()``
    A true one-second-at-a-time generator.  Optionally paced in wall-clock
    time (``speed=1`` is real time, ``speed=60`` is a minute per second,
    ``speed=0`` / ``"max"`` is as fast as the CPU allows).  This is what the
    live animation and the Snowflake notebook consume.

``MarketSimulator.windows()``
    The same tape handed over one *epoch window* (default 1800 s = 30 min) at a
    time as contiguous arrays.  Because every model in this repo is strictly
    causal, feeding a whole window at once is mathematically identical to
    stepping it second by second -- but it runs orders of magnitude faster on a
    GPU.  This is the "train at max speed" path.

The simulator also knows how to shard the tape across distributed ranks: each
rank gets an interleaved set of *blocks* so every rank sees all market regimes
(bull, bear, chop) rather than one contiguous era.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .compact import Bars, ChunkIndex

EPOCH_SECONDS = 1800          # one training epoch == 30 minutes of market time
HORIZON_SECONDS = 1800        # predict 30 minutes ahead


@dataclass
class Tick:
    """One second of market."""
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    taker_buy_base: float
    gap: bool

    @property
    def utc(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(self.ts, dt.timezone.utc)


@dataclass
class Window:
    """One 30-minute epoch window of market, as arrays."""
    t0: int
    bars: Bars
    index: int                      # global window index
    epoch_seconds: int = EPOCH_SECONDS

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def t_end(self) -> int:
        return self.t0 + len(self.bars)

    @property
    def close(self) -> np.ndarray:
        return self.bars.close

    def utc_range(self) -> Tuple[dt.datetime, dt.datetime]:
        f = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc)
        return f(self.t0), f(self.t_end)


class MarketSimulator:
    """
    Replays ``ChunkIndex`` data as a live feed.

    Parameters
    ----------
    index        : ChunkIndex over the compact dataset
    t_from, t_to : epoch-second bounds (defaults to the whole dataset)
    speed        : simulated seconds per wall-clock second.
                   ``0`` or ``"max"`` -> no pacing at all (training mode).
    epoch_seconds: length of one training epoch in simulated seconds.
    warmup       : seconds of history made available *before* t_from so the
                   analyser's long EMAs are already converged at the first tick.
    """

    def __init__(self,
                 index: ChunkIndex,
                 t_from: Optional[int] = None,
                 t_to: Optional[int] = None,
                 speed: float | str = 0,
                 epoch_seconds: int = EPOCH_SECONDS,
                 warmup: int = 7200,
                 prefetch_windows: int = 4):
        self.index = index
        self.t_from = int(t_from if t_from is not None else index.t_start + warmup)
        self.t_to = int(t_to if t_to is not None else index.t_end)
        if self.t_to <= self.t_from:
            raise ValueError(f"empty simulation range [{self.t_from}, {self.t_to})")
        self.speed = 0.0 if (speed == "max" or speed is None) else float(speed)
        self.epoch_seconds = int(epoch_seconds)
        self.warmup = int(warmup)
        self.prefetch_windows = max(1, int(prefetch_windows))
        self._t = self.t_from

    # -- introspection -----------------------------------------------------
    def __repr__(self) -> str:
        f = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
        return (f"<MarketSimulator {f(self.t_from)} -> {f(self.t_to)} "
                f"({self.total_seconds:,} s, {self.n_windows:,} windows of "
                f"{self.epoch_seconds}s, speed={'max' if not self.speed else self.speed})>")

    @property
    def total_seconds(self) -> int:
        return self.t_to - self.t_from

    @property
    def n_windows(self) -> int:
        return self.total_seconds // self.epoch_seconds

    def reset(self) -> None:
        self._t = self.t_from

    # -- warm-up history ---------------------------------------------------
    def history_before(self, ts: int, n: int) -> Bars:
        """``n`` seconds of bars ending just before ``ts`` (for analyser warm-up)."""
        a = max(self.index.t_start, ts - n)
        return self.index.range(a, max(a + 1, ts))

    # -- windowed (fast) interface ----------------------------------------
    def windows(self,
                start_index: int = 0,
                max_windows: Optional[int] = None,
                stride: int = 1) -> Iterator[Window]:
        """
        Yield consecutive epoch windows as arrays.

        Chunks are fetched in batches of ``prefetch_windows`` so the underlying
        ``.npz`` decode cost is amortised over many windows.
        """
        W = self.epoch_seconds
        i = start_index
        produced = 0
        batch = self.prefetch_windows
        while True:
            t0 = self.t_from + i * W * stride
            if t0 + W > self.t_to:
                return
            if max_windows is not None and produced >= max_windows:
                return
            span = min(batch, (self.t_to - t0) // W)
            if max_windows is not None:
                span = min(span, max_windows - produced)
            span = max(1, span)
            big = self.index.range(t0, t0 + span * W)
            for k in range(span):
                a = k * W
                if a + W > len(big):
                    break
                yield Window(t0=t0 + a, bars=big.slice(a, a + W), index=i,
                             epoch_seconds=W)
                i += stride
                produced += 1
                if max_windows is not None and produced >= max_windows:
                    return

    # -- tick (live) interface --------------------------------------------
    def ticks(self,
              t_from: Optional[int] = None,
              t_to: Optional[int] = None,
              speed: Optional[float] = None,
              block: int = 3600) -> Iterator[Tick]:
        """
        Yield one :class:`Tick` per simulated second, optionally wall-clock paced.

        Data is pulled from disk ``block`` seconds at a time, so pacing stays
        smooth even when a chunk boundary is crossed.
        """
        a = int(t_from if t_from is not None else self.t_from)
        b = int(t_to if t_to is not None else self.t_to)
        sp = self.speed if speed is None else float(speed)
        wall0 = time.perf_counter()
        emitted = 0
        t = a
        while t < b:
            n = min(block, b - t)
            bars = self.index.range(t, t + n)
            cl, op, hi, lo = bars.close, bars.open, bars.high, bars.low
            vo, qv, tr, tb, gp = (bars.volume, bars.quote_volume, bars.trades,
                                  bars.taker_buy_base, bars.gap)
            for k in range(len(bars)):
                if sp > 0:
                    target = wall0 + emitted / sp
                    lag = target - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)
                yield Tick(ts=int(bars.t0 + k), open=float(op[k]), high=float(hi[k]),
                           low=float(lo[k]), close=float(cl[k]), volume=float(vo[k]),
                           quote_volume=float(qv[k]), trades=int(tr[k]),
                           taker_buy_base=float(tb[k]), gap=bool(gp[k]))
                emitted += 1
            t += n

    # -- sharding for distributed training ---------------------------------
    def shard(self, rank: int, world_size: int,
              block_windows: int = 8) -> "MarketSimulator":
        """
        Return a view of this simulator restricted to this rank's blocks.

        The tape is cut into blocks of ``block_windows`` epochs and dealt out
        round-robin, so every rank sees the full span of market regimes instead
        of one contiguous (and possibly atypical) era.
        """
        if world_size <= 1:
            return self
        return _ShardedSimulator(self, rank, world_size, block_windows)


class _ShardedSimulator(MarketSimulator):
    """Round-robin block view over a parent simulator (see ``MarketSimulator.shard``)."""

    def __init__(self, parent: MarketSimulator, rank: int, world_size: int,
                 block_windows: int):
        super().__init__(parent.index, parent.t_from, parent.t_to, parent.speed,
                         parent.epoch_seconds, parent.warmup, parent.prefetch_windows)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.block_windows = max(1, int(block_windows))
        self._blocks: List[int] = list(
            range(self.rank * self.block_windows,
                  self.n_windows,
                  self.world_size * self.block_windows))

    def __repr__(self) -> str:
        return (f"<ShardedSimulator rank {self.rank}/{self.world_size} "
                f"{len(self._blocks)} blocks x {self.block_windows} windows>")

    @property
    def n_windows_local(self) -> int:
        return sum(min(self.block_windows, self._total_windows() - b)
                   for b in self._blocks)

    def _total_windows(self) -> int:
        return (self.t_to - self.t_from) // self.epoch_seconds

    def windows(self, start_index: int = 0, max_windows: Optional[int] = None,
                stride: int = 1) -> Iterator[Window]:
        W = self.epoch_seconds
        total = self._total_windows()
        produced = 0
        for b in self._blocks:
            span = min(self.block_windows, total - b)
            if span <= 0:
                continue
            t0 = self.t_from + b * W
            big = self.index.range(t0, t0 + span * W)
            for k in range(span):
                if max_windows is not None and produced >= max_windows:
                    return
                a = k * W
                if a + W > len(big):
                    break
                yield Window(t0=t0 + a, bars=big.slice(a, a + W), index=b + k,
                             epoch_seconds=W)
                produced += 1


# --------------------------------------------------------------------------
def split_train_eval(index: ChunkIndex, eval_fraction: float = 0.05,
                     warmup: int = 7200) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Chronological split -- the eval slice is always the *most recent* data, which
    is the only honest way to evaluate a forecaster.
    """
    t0, t1 = index.t_start + warmup, index.t_end
    span = t1 - t0
    cut = t1 - int(span * eval_fraction)
    cut -= cut % EPOCH_SECONDS
    return (t0, cut), (cut, t1)
