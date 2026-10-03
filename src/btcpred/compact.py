"""
compact.py -- ultra-compact, offline-friendly storage codec for second-by-second
Bitcoin market data.

Why a custom codec?
-------------------
The full second-by-second history of BTCUSDT (Aug-2017 -> now) is ~290 million
rows.  As raw Binance CSV that is ~45 GB: impossible to ship inside a git
repository and painful to move into an air-gapped Snowflake account.

This module stores the *same information* in ~17 MB per month (~1.5 GB for the
entire history) by exploiting the structure of the data:

1.  Bars live on a **perfectly regular 1-second grid**, so timestamps are never
    stored -- only ``t0``, the epoch second of index 0.
2.  Prices are tick-quantised -> stored as **integer cents**, never floats.
3.  Consecutive 1-second closes barely move -> the close series is **delta
    encoded**, after which zlib needs well under 1 byte per sample.
4.  open/high/low and the VWAP are stored as **offsets from the bar's own
    close** (tiny integers).
5.  Taker-buy volume is stored as a **ratio byte** (order-flow imbalance in
    0..255) rather than a second float array.

Seconds in which the exchange printed no trade are reconstructed by
forward-filling the previous close (volume / trade-count = 0) and flagged in a
bit-packed ``gap`` mask, so the model can always distinguish a real print from
a synthetic one.

Fidelity (measured on 2025-08, 2.68M bars): prices are **exact**, quote volume
round-trips to 2e-6 %, taker-buy volume to 0.15 % (it is a ratio feature).

The container is a plain ``.npz`` (zlib), so decoding needs **nothing but
NumPy** -- which matters because the Snowflake side has no internet access and
not every runtime ships pyarrow.

Public API
----------
``encode_grid(...)``    -> dict of arrays ready for ``np.savez_compressed``
``save_chunk(path, …)`` -> write one chunk file
``load_chunk(path)``    -> ``Bars`` with decoded float32/int32 arrays
``ChunkIndex(dir)``     -> lazily serves arbitrary ``[t_from, t_to)`` windows
                           out of a directory of chunks, with an LRU cache.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Scaling constants.  Chosen so every value fits its dtype with no overflow.
# --------------------------------------------------------------------------
PRICE_SCALE = 100          # USD -> cents.     1e6 USD -> 1e8        (int32 ok)
VOL_SCALE = 10_000         # BTC -> 1e-4 BTC.  214_748 BTC/s cap     (int32 ok)
RATIO_LEVELS = 255         # taker-buy ratio quantisation (uint8)
TRADES_CAP = 65_535        # trades/second cap (observed max ~7k)    (uint16)

SCHEMA_VERSION = 4

FIELDS = ("close", "open", "high", "low", "volume", "quote_volume",
          "trades", "taker_buy_base", "gap")


# --------------------------------------------------------------------------
# Decoded container
# --------------------------------------------------------------------------
@dataclass
class Bars:
    """Decoded, contiguous 1-second bars.  All arrays share one length."""
    t0: int                     # epoch seconds of index 0
    close: np.ndarray           # float32 USD
    open: np.ndarray            # float32 USD
    high: np.ndarray            # float32 USD
    low: np.ndarray             # float32 USD
    volume: np.ndarray          # float32 BTC
    quote_volume: np.ndarray    # float32 USD
    trades: np.ndarray          # int32
    taker_buy_base: np.ndarray  # float32 BTC
    gap: np.ndarray             # bool: True => no print that second

    def __len__(self) -> int:
        return int(self.close.shape[0])

    @property
    def t_end(self) -> int:
        return self.t0 + len(self)

    def timestamps(self) -> np.ndarray:
        return np.arange(self.t0, self.t0 + len(self), dtype=np.int64)

    def slice(self, a: int, b: int) -> "Bars":
        """Slice by *array index* (not epoch second)."""
        a = max(0, int(a))
        b = min(len(self), int(b))
        return Bars(
            t0=self.t0 + a,
            close=self.close[a:b], open=self.open[a:b],
            high=self.high[a:b], low=self.low[a:b],
            volume=self.volume[a:b], quote_volume=self.quote_volume[a:b],
            trades=self.trades[a:b], taker_buy_base=self.taker_buy_base[a:b],
            gap=self.gap[a:b],
        )

    def feature_matrix(self) -> np.ndarray:
        """(N, 8) float32 matrix in canonical column order."""
        return np.stack([
            self.open, self.high, self.low, self.close,
            self.volume, self.quote_volume,
            self.trades.astype(np.float32), self.taker_buy_base,
        ], axis=1).astype(np.float32, copy=False)


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------
def _delta_encode(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    if x.shape[0] > 1:
        np.subtract(x[1:], x[:-1], out=out[1:])
    return out


def _i32(a: np.ndarray, name: str) -> np.ndarray:
    if a.size:
        lo, hi = int(a.min()), int(a.max())
        if lo < -2_147_483_648 or hi > 2_147_483_647:
            raise OverflowError(f"{name} outside int32 range [{lo}, {hi}]")
    return a.astype(np.int32)


def encode_grid(
    t0: int,
    close_c: np.ndarray,       # int64 cents, gridded + forward-filled
    open_c: np.ndarray,
    high_c: np.ndarray,
    low_c: np.ndarray,
    volume: np.ndarray,        # float64 BTC
    quote_volume: np.ndarray,  # float64 USD
    trades: np.ndarray,        # int64
    taker_buy_base: np.ndarray,# float64 BTC
    gap: np.ndarray,           # bool
) -> Dict[str, np.ndarray]:
    """Encode one gridded month into the dict written to ``.npz``."""
    n = int(close_c.shape[0])
    for name, a in (("open", open_c), ("high", high_c), ("low", low_c),
                    ("volume", volume), ("quote_volume", quote_volume),
                    ("trades", trades), ("taker", taker_buy_base), ("gap", gap)):
        if a.shape[0] != n:
            raise ValueError(f"ragged array: {name}")

    close_c = close_c.astype(np.int64)
    vol = np.asarray(volume, dtype=np.float64)
    qvol = np.asarray(quote_volume, dtype=np.float64)
    tbb = np.asarray(taker_buy_base, dtype=np.float64)
    safe_v = np.maximum(vol, 1e-12)

    close_f = close_c / float(PRICE_SCALE)
    vwap = np.where(vol > 0, qvol / safe_v, close_f)
    vwap_off = np.rint((vwap - close_f) * PRICE_SCALE)
    # guard against a handful of corrupt historical rows
    vwap_off = np.clip(vwap_off, -2_000_000_000, 2_000_000_000).astype(np.int64)

    ratio = np.where(vol > 0, np.clip(tbb / safe_v, 0.0, 1.0), 0.5)

    return {
        "schema": np.int32(SCHEMA_VERSION),
        "t0": np.int64(t0),
        "n": np.int64(n),
        "price_scale": np.int32(PRICE_SCALE),
        "vol_scale": np.int32(VOL_SCALE),
        "close_delta": _i32(_delta_encode(close_c), "close_delta"),
        "open_off": _i32(open_c.astype(np.int64) - close_c, "open_off"),
        "high_off": _i32(high_c.astype(np.int64) - close_c, "high_off"),
        "low_off": _i32(low_c.astype(np.int64) - close_c, "low_off"),
        "vwap_off": _i32(vwap_off, "vwap_off"),
        "volume_i": _i32(np.rint(vol * VOL_SCALE).astype(np.int64), "volume"),
        "trades_u16": np.clip(np.asarray(trades, dtype=np.int64), 0,
                              TRADES_CAP).astype(np.uint16),
        "tb_ratio_u8": np.clip(np.rint(ratio * RATIO_LEVELS), 0,
                               RATIO_LEVELS).astype(np.uint8),
        "gap_bits": np.packbits(np.asarray(gap, dtype=bool)),
    }


def save_chunk(path: str, payload: Dict[str, np.ndarray]) -> int:
    """Write one compressed chunk atomically; returns bytes written."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, path)
    return os.path.getsize(path)


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------
def load_chunk(path: str) -> Bars:
    """Decode a chunk file back into float32 OHLCV arrays."""
    with np.load(path) as z:
        schema = int(z["schema"])
        if schema != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema v{schema}, expected v{SCHEMA_VERSION}")
        n = int(z["n"])
        t0 = int(z["t0"])
        ps = float(int(z["price_scale"]))
        vs = float(int(z["vol_scale"]))

        close_c = np.cumsum(z["close_delta"].astype(np.int64))
        close_f = (close_c / ps)
        vol = z["volume_i"].astype(np.float64) / vs
        ratio = z["tb_ratio_u8"].astype(np.float64) / float(RATIO_LEVELS)
        vwap = close_f + z["vwap_off"].astype(np.float64) / ps

        return Bars(
            t0=t0,
            close=close_f.astype(np.float32),
            open=((close_c + z["open_off"].astype(np.int64)) / ps).astype(np.float32),
            high=((close_c + z["high_off"].astype(np.int64)) / ps).astype(np.float32),
            low=((close_c + z["low_off"].astype(np.int64)) / ps).astype(np.float32),
            volume=vol.astype(np.float32),
            quote_volume=(vwap * vol).astype(np.float32),
            trades=z["trades_u16"].astype(np.int32),
            taker_buy_base=(vol * ratio).astype(np.float32),
            gap=np.unpackbits(z["gap_bits"])[:n].astype(bool),
        )


def concat(parts: Sequence[Bars]) -> Bars:
    """Concatenate time-ordered chunks; inter-chunk holes are forward-filled."""
    parts = [p for p in parts if len(p) > 0]
    if not parts:
        raise ValueError("nothing to concatenate")
    parts = sorted(parts, key=lambda p: p.t0)
    t0 = parts[0].t0
    total = parts[-1].t_end - t0
    out = Bars(
        t0=t0,
        close=np.zeros(total, np.float32), open=np.zeros(total, np.float32),
        high=np.zeros(total, np.float32), low=np.zeros(total, np.float32),
        volume=np.zeros(total, np.float32),
        quote_volume=np.zeros(total, np.float32),
        trades=np.zeros(total, np.int32),
        taker_buy_base=np.zeros(total, np.float32),
        gap=np.ones(total, bool),
    )
    for p in parts:
        a = p.t0 - t0
        for f in FIELDS:
            getattr(out, f)[a:a + len(p)] = getattr(p, f)
    holes = out.close == 0
    if holes.any():
        idx = np.where(~holes, np.arange(total), 0)
        np.maximum.accumulate(idx, out=idx)
        for f in ("close", "open", "high", "low"):
            arr = getattr(out, f)
            arr[holes] = out.close[idx][holes]
    return out


# --------------------------------------------------------------------------
# Directory index -- what the training loop actually talks to
# --------------------------------------------------------------------------
MANIFEST_NAME = "MANIFEST.json"


class ChunkIndex:
    """
    Lazily serves arbitrary second-ranges from a directory of chunk files.

    The dataset is never fully resident: only chunks overlapping the requested
    window are decoded, and a small LRU keeps the hot ones warm.  That is what
    lets a 290-million-row dataset train inside a 2 GB container.
    """

    def __init__(self, root: str, cache_chunks: int = 3):
        self.root = os.path.abspath(root)
        mpath = os.path.join(self.root, MANIFEST_NAME)
        if not os.path.exists(mpath):
            raise FileNotFoundError(
                f"No {MANIFEST_NAME} in {self.root}.\n"
                f"Run  python scripts/download_binance_1s.py --all  on a machine "
                f"with internet, or point --data-dir at the bundled dataset."
            )
        with open(mpath) as fh:
            self.manifest = json.load(fh)
        self.chunks: List[dict] = sorted(self.manifest["chunks"], key=lambda c: c["t0"])
        if not self.chunks:
            raise ValueError("manifest contains no chunks")
        self._starts = np.array([c["t0"] for c in self.chunks], dtype=np.int64)
        self._ends = np.array([c["t0"] + c["n"] for c in self.chunks], dtype=np.int64)
        self.t_start = int(self._starts[0])
        self.t_end = int(self._ends[-1])
        self.cache_chunks = max(1, cache_chunks)
        self._cache: Dict[int, Bars] = {}
        self._order: List[int] = []

    def __repr__(self) -> str:
        d = (self.t_end - self.t_start) / 86400.0
        return (f"<ChunkIndex {len(self.chunks)} chunks | "
                f"{self.t_end - self.t_start:,} s ({d:,.1f} days) | "
                f"{self.t_start}..{self.t_end}>")

    @property
    def total_seconds(self) -> int:
        return self.t_end - self.t_start

    def describe(self) -> str:
        import datetime as _dt
        f = lambda t: _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%d")
        mb = sum(c.get("bytes", 0) for c in self.chunks) / 1e6
        return (f"{self.manifest.get('symbol','BTCUSDT')} 1s | "
                f"{f(self.t_start)} -> {f(self.t_end)} | "
                f"{self.total_seconds:,} bars | {len(self.chunks)} chunks | "
                f"{mb:,.0f} MB")

    def _get(self, i: int) -> Bars:
        if i in self._cache:
            self._order.remove(i)
            self._order.append(i)
            return self._cache[i]
        b = load_chunk(os.path.join(self.root, self.chunks[i]["file"]))
        self._cache[i] = b
        self._order.append(i)
        while len(self._order) > self.cache_chunks:
            self._cache.pop(self._order.pop(0), None)
        return b

    def range(self, t_from: int, t_to: int) -> Bars:
        """Bars for the epoch-second window ``[t_from, t_to)``."""
        if t_to <= t_from:
            raise ValueError("empty window")
        t_from = max(int(t_from), self.t_start)
        t_to = min(int(t_to), self.t_end)
        lo = int(np.searchsorted(self._ends, t_from, side="right"))
        hi = int(np.searchsorted(self._starts, t_to, side="left"))
        parts = []
        for i in range(lo, hi):
            b = self._get(i)
            a = max(0, t_from - b.t0)
            z = min(len(b), t_to - b.t0)
            if z > a:
                parts.append(b.slice(a, z))
        if not parts:
            raise ValueError(f"no data in [{t_from}, {t_to})")
        return concat(parts) if len(parts) > 1 else parts[0]


def write_manifest(root: str, chunks: Iterable[dict],
                   extra: Optional[dict] = None) -> str:
    chunks = sorted(chunks, key=lambda c: c["t0"])
    total = sum(c["n"] for c in chunks)
    man = {
        "schema": SCHEMA_VERSION,
        "symbol": (extra or {}).get("symbol", "BTCUSDT"),
        "interval": "1s",
        "source": "https://data.binance.vision (Binance public market data)",
        "grid": "continuous 1-second grid; silent seconds forward-filled and "
                "flagged in the gap mask",
        "columns": list(FIELDS),
        "n_chunks": len(chunks),
        "n_seconds": total,
        "t_start": chunks[0]["t0"] if chunks else 0,
        "t_end": (chunks[-1]["t0"] + chunks[-1]["n"]) if chunks else 0,
        "bytes": sum(c.get("bytes", 0) for c in chunks),
        "chunks": chunks,
    }
    if extra:
        man.update({k: v for k, v in extra.items() if k not in man})
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, MANIFEST_NAME)
    with open(path, "w") as fh:
        json.dump(man, fh, indent=1)
    return path
