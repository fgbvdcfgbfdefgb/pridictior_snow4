#!/usr/bin/env python3
"""
download_binance_1s.py -- build the full second-by-second BTC dataset.

Downloads Binance's public 1-second kline archives, snaps every month onto a
continuous 1-second grid, and writes it with the compact codec in
``btcpred.compact`` (~17 MB per month instead of ~400 MB of CSV).

THIS IS THE ONLY SCRIPT THAT NEEDS INTERNET.  Run it once on a connected
machine; everything else in this repository -- all training, evaluation and the
Snowflake notebook -- then runs fully offline against ``data/btcusdt_1s/``.

It is deliberately **memory-frugal**: archives stream to disk, CSVs are parsed
in row-chunks, and the per-month grid is accumulated in int32/float32.  The
whole 2017->today history builds inside a 2 GB container.

Examples
--------
    python scripts/download_binance_1s.py --all                 # whole history
    python scripts/download_binance_1s.py --start 2024-01 --end 2025-12
    python scripts/download_binance_1s.py --all --resume        # top-up only
    python scripts/download_binance_1s.py --reindex-only        # rebuild manifest
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import io
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from btcpred.compact import (PRICE_SCALE, encode_grid, save_chunk,  # noqa: E402
                             write_manifest)

MONTHLY = "https://data.binance.vision/data/spot/monthly/klines"
DAILY = "https://data.binance.vision/data/spot/daily/klines"
FIRST_MONTH = (2017, 8)
USER_AGENT = "btcpred-dataset-builder/1.0"
ROWS_PER_CHUNK = 1_000_000


# --------------------------------------------------------------------------
# calendar helpers
# --------------------------------------------------------------------------
def months_between(start, end):
    y, m = start
    out = []
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def parse_month(s: str):
    y, m = s.split("-")
    return int(y), int(m)


def month_bounds(y: int, m: int):
    a = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
    b = (dt.datetime(y + 1, 1, 1, tzinfo=dt.timezone.utc) if m == 12
         else dt.datetime(y, m + 1, 1, tzinfo=dt.timezone.utc))
    return int(a.timestamp()), int(b.timestamp())


# --------------------------------------------------------------------------
# network -- stream straight to disk, never into RAM
# --------------------------------------------------------------------------
def download_to(url: str, path: str, retries: int = 4, timeout: int = 300) -> bool:
    """Returns True if downloaded, False on 404.  Raises on repeated failure."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r, \
                    open(path, "wb") as fh:
                shutil.copyfileobj(r, fh, length=1 << 20)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))
    return False


def extract_csv(zip_path: str, csv_path: str, append: bool = False) -> bool:
    """Extract the single CSV member of a Binance archive to ``csv_path``."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                return False
            with z.open(names[0]) as src, open(csv_path, "ab" if append else "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
        return True
    except zipfile.BadZipFile:
        return False


def fetch_month_csv(symbol: str, y: int, m: int, workdir: str) -> str | None:
    """
    Materialise one month of klines as a CSV file on disk.
    Prefers the monthly archive; falls back to stitching daily archives
    (needed for the current / most recent month, which has no monthly file yet).
    """
    csv_path = os.path.join(workdir, f"{symbol}-{y:04d}-{m:02d}.csv")
    zip_path = csv_path + ".zip"
    url = f"{MONTHLY}/{symbol}/1s/{symbol}-1s-{y:04d}-{m:02d}.zip"
    try:
        if download_to(url, zip_path):
            ok = extract_csv(zip_path, csv_path)
            os.remove(zip_path)
            if ok:
                return csv_path
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # daily fallback
    got = False
    d = dt.date(y, m, 1)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    while d.month == m:
        durl = f"{DAILY}/{symbol}/1s/{symbol}-1s-{d:%Y-%m-%d}.zip"
        try:
            if download_to(durl, zip_path):
                if extract_csv(zip_path, csv_path, append=True):
                    got = True
        except Exception:
            pass
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        d += dt.timedelta(days=1)
    return csv_path if got else None


# --------------------------------------------------------------------------
# CSV -> dense 1-second grid (chunked, low memory)
# --------------------------------------------------------------------------
USECOLS = [0, 1, 2, 3, 4, 5, 7, 8, 9]   # t, o, h, l, c, v, quote_vol, n, tbb
NAMES = ["t", "o", "h", "l", "c", "v", "qv", "nt", "tbb"]


def _iter_rows(csv_path: str):
    """Yield (N, 9) float64 blocks from a Binance kline CSV."""
    with open(csv_path, "rb") as fh:
        head = fh.read(256).decode("utf-8", "replace")
    has_header = head[:1].isalpha()
    try:
        import pandas as pd
        reader = pd.read_csv(csv_path, header=0 if has_header else None,
                             usecols=USECOLS, names=None if has_header else NAMES,
                             dtype=np.float64, engine="c",
                             chunksize=ROWS_PER_CHUNK, on_bad_lines="skip")
        for df in reader:
            if has_header:
                df = df.iloc[:, :9]
            yield df.to_numpy(dtype=np.float64, copy=False)
    except ImportError:
        with open(csv_path, "rb") as fh:
            if has_header:
                fh.readline()
            buf = []
            for line in fh:
                p = line.split(b",")
                if len(p) < 10:
                    continue
                buf.append([float(p[i]) for i in USECOLS])
                if len(buf) >= ROWS_PER_CHUNK:
                    yield np.asarray(buf, dtype=np.float64)
                    buf = []
            if buf:
                yield np.asarray(buf, dtype=np.float64)


def grid_month(csv_path: str, y: int, m: int):
    """Snap one month of klines onto a dense 1-second grid (int32/float32)."""
    t_start, t_end = month_bounds(y, m)
    n = t_end - t_start

    close_c = np.zeros(n, np.int32)
    open_c = np.zeros(n, np.int32)
    high_c = np.zeros(n, np.int32)
    low_c = np.zeros(n, np.int32)
    vol = np.zeros(n, np.float32)
    qvol = np.zeros(n, np.float32)
    ntr = np.zeros(n, np.int32)
    tbb = np.zeros(n, np.float32)
    present = np.zeros(n, np.bool_)

    any_row = False
    for arr in _iter_rows(csv_path):
        if arr.size == 0:
            continue
        ot = arr[:, 0]
        # open_time is milliseconds in older archives, microseconds in newer
        div = 1_000_000.0 if np.nanmedian(ot) > 1e14 else 1_000.0
        idx = np.floor(ot / div).astype(np.int64) - t_start
        ok = (idx >= 0) & (idx < n)
        if not ok.any():
            continue
        idx = idx[ok]
        a = arr[ok]
        any_row = True
        close_c[idx] = np.rint(a[:, 4] * PRICE_SCALE).astype(np.int64)
        open_c[idx] = np.rint(a[:, 1] * PRICE_SCALE).astype(np.int64)
        high_c[idx] = np.rint(a[:, 2] * PRICE_SCALE).astype(np.int64)
        low_c[idx] = np.rint(a[:, 3] * PRICE_SCALE).astype(np.int64)
        vol[idx] = a[:, 5]
        qvol[idx] = a[:, 6]
        ntr[idx] = a[:, 7].astype(np.int64)
        tbb[idx] = a[:, 8]
        present[idx] = True
        del arr, a, idx, ok

    if not any_row:
        return None

    # Trim synthetic padding at the edges.  The first month a symbol traded
    # starts part-way through, and the current month is still being written:
    # without this the dataset would end in weeks of forward-filled flat line
    # and poison the (chronological) hold-out split.
    real = np.flatnonzero(present)
    lo, hi = int(real[0]), int(real[-1]) + 1
    if lo > 0 or hi < n:
        head, tail = lo, n - hi
        t_start += lo
        n = hi - lo
        sl = slice(lo, hi)
        close_c, open_c, high_c, low_c = (close_c[sl], open_c[sl],
                                          high_c[sl], low_c[sl])
        vol, qvol, ntr, tbb = vol[sl], qvol[sl], ntr[sl], tbb[sl]
        present = present[sl]
        print(f"[dataset]    {y}-{m:02d}: trimmed {head:,} leading and "
              f"{tail:,} trailing padding seconds -> {n:,} real seconds",
              flush=True)

    # forward-fill the close through silent seconds, back-fill the head
    fill = np.where(present, np.arange(n, dtype=np.int64), 0)
    np.maximum.accumulate(fill, out=fill)
    first = int(np.argmax(present))
    close_c = close_c[fill]
    if first > 0:
        close_c[:first] = close_c[first]
    del fill
    gap = ~present
    open_c[gap] = close_c[gap]
    high_c[gap] = close_c[gap]
    low_c[gap] = close_c[gap]

    return encode_grid(t_start, close_c.astype(np.int64), open_c.astype(np.int64),
                       high_c.astype(np.int64), low_c.astype(np.int64),
                       vol.astype(np.float64), qvol.astype(np.float64),
                       ntr.astype(np.int64), tbb.astype(np.float64), gap)


# --------------------------------------------------------------------------
def rebuild_manifest(out: str, symbol: str = "BTCUSDT") -> dict:
    """(Re)generate MANIFEST.json by scanning the chunk directory."""
    chunks = []
    for fn in sorted(os.listdir(out)):
        if not fn.endswith(".npz"):
            continue
        p = os.path.join(out, fn)
        try:
            with np.load(p) as z:
                chunks.append({"file": fn, "t0": int(z["t0"]), "n": int(z["n"]),
                               "bytes": os.path.getsize(p)})
        except Exception as e:                       # noqa: BLE001
            print(f"[dataset]  !! unreadable chunk {fn}: {e}", flush=True)
    write_manifest(out, chunks, {"symbol": symbol})
    tot = sum(c["n"] for c in chunks)
    mb = sum(c["bytes"] for c in chunks) / 1e6
    print(f"[dataset] manifest: {len(chunks)} chunks | {tot:,} seconds "
          f"({tot/86400:,.0f} days) | {mb:,.0f} MB", flush=True)
    return {"chunks": len(chunks), "seconds": tot, "mb": mb}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default=None, help="YYYY-MM (default 2017-08)")
    ap.add_argument("--end", default=None, help="YYYY-MM (default: current month)")
    ap.add_argument("--all", action="store_true", help="whole published history")
    ap.add_argument("--out", default="data/btcusdt_1s")
    ap.add_argument("-j", "--download-threads", type=int, default=2)
    ap.add_argument("-w", "--workers", type=int, default=1,
                    help="CSV->grid converter threads (memory-hungry; 1-2)")
    ap.add_argument("--workdir", default=None, help="scratch dir for archives")
    ap.add_argument("--resume", action="store_true", help="skip existing months")
    ap.add_argument("--reindex-only", action="store_true",
                    help="just rebuild MANIFEST.json and exit")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.reindex_only:
        rebuild_manifest(args.out, args.symbol)
        return 0

    today = dt.datetime.now(dt.timezone.utc).date()
    start = parse_month(args.start) if args.start else FIRST_MONTH
    end = parse_month(args.end) if args.end else (today.year, today.month)
    if args.all:
        start, end = FIRST_MONTH, (today.year, today.month)

    todo = months_between(start, end)
    if args.resume:
        todo = [(y, m) for (y, m) in todo
                if not os.path.exists(os.path.join(args.out, f"{y:04d}-{m:02d}.npz"))]

    print(f"[dataset] symbol={args.symbol} months={len(todo)} "
          f"({start[0]}-{start[1]:02d} .. {end[0]}-{end[1]:02d}) -> {args.out}",
          flush=True)
    if not todo:
        rebuild_manifest(args.out, args.symbol)
        return 0

    workdir = args.workdir or tempfile.mkdtemp(prefix="btc1s_")
    os.makedirs(workdir, exist_ok=True)
    q: "queue.Queue" = queue.Queue(maxsize=max(1, args.workers))
    state = {"n": 0, "bytes": 0}
    lock = threading.Lock()
    t_begin = time.time()

    def producer(ym):
        y, m = ym
        try:
            p = fetch_month_csv(args.symbol, y, m, workdir)
        except Exception as e:                        # noqa: BLE001
            print(f"[dataset]  !! {y}-{m:02d} download failed: {e}", flush=True)
            return
        if p is None:
            print(f"[dataset]  -- {y}-{m:02d} not published, skipped", flush=True)
            return
        q.put((ym, p))                                # blocks -> back-pressure

    def consumer():
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                return
            (y, m), csv_path = item
            try:
                payload = grid_month(csv_path, y, m)
                if payload is not None:
                    dest = os.path.join(args.out, f"{y:04d}-{m:02d}.npz")
                    sz = save_chunk(dest, payload)
                    del payload
                    with lock:
                        state["n"] += 1
                        state["bytes"] += sz
                        el = time.time() - t_begin
                        rate = state["n"] / max(el, 1e-9)
                        eta = (len(todo) - state["n"]) / max(rate, 1e-9)
                        print(f"[dataset] ok {y}-{m:02d} {sz/1e6:6.1f} MB "
                              f"[{state['n']}/{len(todo)}] "
                              f"{state['bytes']/1e6:,.0f} MB total "
                              f"{el:.0f}s elapsed, ETA {eta/60:.1f} min",
                              flush=True)
            except Exception as e:                    # noqa: BLE001
                print(f"[dataset]  !! {y}-{m:02d} convert failed: {e!r}", flush=True)
            finally:
                try:
                    os.remove(csv_path)
                except OSError:
                    pass
                q.task_done()

    cons = [threading.Thread(target=consumer, daemon=True) for _ in range(args.workers)]
    for c in cons:
        c.start()
    with cf.ThreadPoolExecutor(max_workers=args.download_threads) as ex:
        list(ex.map(producer, todo))
    q.join()
    for _ in cons:
        q.put(None)
    for c in cons:
        c.join()

    rebuild_manifest(args.out, args.symbol)
    if not args.workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    print(f"[dataset] DONE in {(time.time()-t_begin)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
