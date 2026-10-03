# `data/btcusdt_1s/` — the second-by-second dataset

**111 chunk files · 288,043,172 one-second bars · 2017-08-17 → 2026-10-03 ·
2.35 GB**

One `.npz` per calendar month plus `MANIFEST.json`. This is the complete
published history of Binance 1-second BTCUSDT klines, re-encoded so it fits in
a git repository and decodes with nothing but NumPy — which is what lets
training run inside an air-gapped Snowflake account.

## Reading it

```python
import sys; sys.path.insert(0, "src")
from btcpred.compact import ChunkIndex

idx = ChunkIndex("data/btcusdt_1s")
print(idx.describe())
# BTCUSDT 1s | 2017-08-17 -> 2026-10-03 | 288,043,172 bars | 111 chunks | 2,350 MB

bars = idx.range(1_700_000_000, 1_700_001_800)   # any [t_from, t_to) in epoch s
bars.close, bars.volume, bars.taker_buy_base, bars.gap
```

Only the chunks overlapping the requested window are decoded, and a small LRU
keeps the hot ones warm — a 288-million-row dataset streams fine inside a 2 GB
container.

## Columns

| field | dtype | meaning |
|---|---|---|
| `close` `open` `high` `low` | float32 | USD, exact on the cent grid |
| `volume` | float32 | BTC traded that second |
| `quote_volume` | float32 | USD traded that second |
| `trades` | int32 | number of trades that second |
| `taker_buy_base` | float32 | BTC bought by takers (order-flow imbalance) |
| `gap` | bool | **True = no trade printed that second**; price forward-filled |

Timestamps are implicit: bar *i* is epoch second `t0 + i`.

## How 45 GB became 2.35 GB

| trick | effect |
|---|---|
| no timestamp column — bars lie on a regular 1-second grid | −8 B/row |
| prices as integer **cents**, not floats | exact *and* compressible |
| close **delta encoded** before zlib | < 1 B/sample |
| open/high/low/VWAP as **offsets from the bar's own close** | tiny integers |
| taker flow as a **ratio byte** instead of a second float array | −3.4 MB/month |
| `gap` mask **bit-packed** | −2.6 MB/month |

Measured fidelity on 2.68 M bars: prices **exact**, quote volume round-trips to
2e-6 %, taker-buy volume to 0.15 % (it is consumed as a ratio anyway).

## Edges are trimmed

BTCUSDT's first trade was **2017-08-17 04:00 UTC**, and the newest month is
still being written. Both partial months are cut to their real data instead of
being padded — otherwise the chronological hold-out split (the newest 5 % of
the tape) would be scored against weeks of forward-filled flat line.

## Rebuilding / extending

Only `scripts/download_binance_1s.py` touches the network.

```bash
python scripts/download_binance_1s.py --all            # everything
python scripts/download_binance_1s.py --all --resume   # add new months
python scripts/download_binance_1s.py --start 2025-01 --end 2025-12
python scripts/verify_dataset.py --deep                # decode every chunk
```

The builder streams archives to disk, parses CSV in row-chunks and accumulates
the month in int32/float32, so it completes inside a 2 GB container.

Source: <https://data.binance.vision> · data © Binance, subject to their terms.
