"""
storage.py -- persist *everything*.

The brief: "every 30-minute window is checkpointed; every per-second prediction
within that window is stored and used to compute the reward/loss signal for the
next training cycle", with a petabyte of storage available -- so nothing is
thrown away.

Layout of a run
---------------
    runs/<run_id>/
        run.json                 config, hardware profile, git-ish provenance
        metrics.csv              one row per window (append-only, human readable)
        predictions/
            w00000000.npz        per-second record for window 0
            w00000001.npz        ...
        checkpoints/
            ckpt_00000000.npz    model + analyser + filter + normaliser state
            latest.json          pointer to the newest checkpoint
        live/
            state.json           tiny file the live dashboard polls
            frames/              optional PNG frames of the animation

Each ``predictions/wNNNNNNNN.npz`` holds, for all 1800 seconds of the window:
``ts``, ``price_now``, ``mu_raw``, ``mu_smooth``, ``sigma``, ``pred_price``,
and -- once the horizon has elapsed -- ``true_future_price`` and
``target_permille``.  That is the reward/loss tape of the whole run.

Writes are atomic (tmp + rename) so a run can be killed at any instant and the
store stays readable.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _atomic_write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _atomic_write_text(path: str, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


class RunStore:
    """Filesystem-backed store for one training run."""

    METRIC_FIELDS = [
        "window", "wall_time", "t0", "t_end", "utc", "rank",
        "loss", "acc_loss", "nll", "dir_loss", "smooth", "curve",
        "mae_permille", "rmse_permille", "mape_pct", "band_accuracy",
        "direction_acc", "r2", "skill_vs_naive", "reward", "bias_permille",
        "jitter_permille", "max_slew_permille", "sigma_mean", "calibration",
        "grad_norm", "lr", "sim_sec_per_sec",
    ]

    def __init__(self, root: str, run_id: Optional[str] = None,
                 rank: int = 0, resume: bool = False):
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.root = os.path.abspath(os.path.join(root, self.run_id))
        self.rank = int(rank)
        self.pred_dir = os.path.join(self.root, "predictions")
        self.ckpt_dir = os.path.join(self.root, "checkpoints")
        self.live_dir = os.path.join(self.root, "live")
        for d in (self.root, self.pred_dir, self.ckpt_dir, self.live_dir):
            os.makedirs(d, exist_ok=True)
        self.metrics_path = os.path.join(
            self.root, "metrics.csv" if rank == 0 else f"metrics_rank{rank}.csv")
        if not os.path.exists(self.metrics_path) or not resume:
            with open(self.metrics_path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self.METRIC_FIELDS).writeheader()
        self._rows = 0

    # -- run-level metadata -------------------------------------------------
    def write_run_json(self, payload: Dict[str, Any]) -> None:
        if self.rank != 0:
            return
        payload = dict(payload)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
        _atomic_write_text(os.path.join(self.root, "run.json"),
                           json.dumps(payload, indent=2, default=str))

    # -- per-window prediction tape ------------------------------------------
    def save_window(self, window_index: int, ts: np.ndarray, price_now: np.ndarray,
                    mu_raw: np.ndarray, mu_smooth: np.ndarray, sigma: np.ndarray,
                    pred_price: np.ndarray,
                    true_future_price: Optional[np.ndarray] = None,
                    target_permille: Optional[np.ndarray] = None,
                    extra: Optional[Dict[str, np.ndarray]] = None) -> str:
        """Store all per-second predictions of one window (float32, compressed)."""
        payload = {
            "window": np.int64(window_index),
            "rank": np.int32(self.rank),
            "ts": np.asarray(ts, np.int64),
            "price_now": np.asarray(price_now, np.float32),
            "mu_raw": np.asarray(mu_raw, np.float32),
            "mu_smooth": np.asarray(mu_smooth, np.float32),
            "sigma": np.asarray(sigma, np.float32),
            "pred_price": np.asarray(pred_price, np.float32),
        }
        if true_future_price is not None:
            payload["true_future_price"] = np.asarray(true_future_price, np.float32)
        if target_permille is not None:
            payload["target_permille"] = np.asarray(target_permille, np.float32)
        if extra:
            for k, v in extra.items():
                payload[k] = np.asarray(v)
        name = (f"w{window_index:08d}.npz" if self.rank == 0
                else f"w{window_index:08d}_r{self.rank}.npz")
        path = os.path.join(self.pred_dir, name)
        buf = io.BytesIO()
        np.savez_compressed(buf, **payload)
        _atomic_write_bytes(path, buf.getvalue())
        return path

    def load_window(self, window_index: int) -> Dict[str, np.ndarray]:
        name = (f"w{window_index:08d}.npz" if self.rank == 0
                else f"w{window_index:08d}_r{self.rank}.npz")
        with np.load(os.path.join(self.pred_dir, name)) as z:
            return {k: z[k] for k in z.files}

    def iter_windows(self) -> Iterable[Dict[str, np.ndarray]]:
        for fn in sorted(os.listdir(self.pred_dir)):
            if fn.endswith(".npz"):
                with np.load(os.path.join(self.pred_dir, fn)) as z:
                    yield {k: z[k] for k in z.files}

    # -- metrics ---------------------------------------------------------------
    def append_metrics(self, row: Dict[str, Any]) -> None:
        out = {k: row.get(k, "") for k in self.METRIC_FIELDS}
        out["rank"] = self.rank
        with open(self.metrics_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.METRIC_FIELDS).writerow(out)
        self._rows += 1

    def read_metrics(self) -> List[Dict[str, str]]:
        with open(self.metrics_path) as fh:
            return list(csv.DictReader(fh))

    # -- checkpoints ------------------------------------------------------------
    def save_checkpoint(self, window_index: int, blob: Dict[str, Any],
                        keep_last: int = 0) -> str:
        """
        One checkpoint per 30-minute window.

        Stored as ``.npz`` of pickled components so a checkpoint can be read
        back without torch present (the torch tensors are converted to NumPy
        before saving).
        """
        path = os.path.join(self.ckpt_dir, f"ckpt_{window_index:08d}.npz")
        flat = _flatten_for_npz(blob)
        buf = io.BytesIO()
        np.savez_compressed(buf, **flat)
        _atomic_write_bytes(path, buf.getvalue())
        _atomic_write_text(os.path.join(self.ckpt_dir, "latest.json"),
                           json.dumps({"window": window_index,
                                       "file": os.path.basename(path),
                                       "time": time.time()}, indent=1))
        if keep_last > 0:
            self.prune_checkpoints(keep_last)
        return path

    def prune_checkpoints(self, keep_last: int) -> int:
        files = sorted(f for f in os.listdir(self.ckpt_dir)
                       if f.startswith("ckpt_") and f.endswith(".npz"))
        drop = files[:-keep_last] if keep_last < len(files) else []
        for f in drop:
            try:
                os.remove(os.path.join(self.ckpt_dir, f))
            except OSError:
                pass
        return len(drop)

    def latest_checkpoint(self) -> Optional[str]:
        p = os.path.join(self.ckpt_dir, "latest.json")
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            meta = json.load(fh)
        full = os.path.join(self.ckpt_dir, meta["file"])
        return full if os.path.exists(full) else None

    def load_checkpoint(self, path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        path = path or self.latest_checkpoint()
        if not path or not os.path.exists(path):
            return None
        with np.load(path, allow_pickle=True) as z:
            return _unflatten_from_npz({k: z[k] for k in z.files})

    # -- live dashboard feed -------------------------------------------------------
    def write_live_state(self, state: Dict[str, Any]) -> None:
        if self.rank != 0:
            return
        _atomic_write_text(os.path.join(self.live_dir, "state.json"),
                           json.dumps(state, default=_json_default))

    # -- housekeeping ---------------------------------------------------------------
    def size_bytes(self) -> int:
        tot = 0
        for d, _, fs in os.walk(self.root):
            for f in fs:
                try:
                    tot += os.path.getsize(os.path.join(d, f))
                except OSError:
                    pass
        return tot

    def summary(self) -> str:
        n_pred = len([f for f in os.listdir(self.pred_dir) if f.endswith(".npz")])
        n_ck = len([f for f in os.listdir(self.ckpt_dir) if f.endswith(".npz")])
        return (f"run {self.run_id}: {n_pred} prediction windows, "
                f"{n_ck} checkpoints, {self.size_bytes()/1e6:.1f} MB at {self.root}")


# --------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _flatten_for_npz(blob: Dict[str, Any], prefix: str = "") -> Dict[str, np.ndarray]:
    """Turn a nested dict of arrays / tensors / scalars into a flat npz payload."""
    out: Dict[str, np.ndarray] = {}
    for k, v in blob.items():
        key = f"{prefix}{k}"
        if hasattr(v, "detach"):                        # torch tensor
            out[key] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            out[key] = v
        elif isinstance(v, dict):
            out.update(_flatten_for_npz(v, prefix=key + "/"))
        elif isinstance(v, (int, float, bool, np.number)):
            out[key] = np.asarray(v)
        elif v is None:
            out[key] = np.asarray("__none__")
        else:
            out[key] = np.asarray(json.dumps(v, default=_json_default))
    return out


def _unflatten_from_npz(flat: Dict[str, np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in flat.items():
        parts = k.split("/")
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        if v.dtype.kind in "US":
            s = str(v)
            if s == "__none__":
                d[parts[-1]] = None
                continue
            try:
                d[parts[-1]] = json.loads(s)
                continue
            except Exception:
                d[parts[-1]] = s
                continue
        d[parts[-1]] = v
    return out


def consolidate_predictions(run_dir: str, out_path: Optional[str] = None) -> str:
    """
    Merge every per-window prediction file into one flat table.

    Writes Parquet when pyarrow is importable (best for Snowflake ingestion),
    otherwise gzipped CSV -- both are offline-friendly.
    """
    pred_dir = os.path.join(run_dir, "predictions")
    files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"no prediction windows in {pred_dir}")
    cols: Dict[str, List[np.ndarray]] = {}
    for f in files:
        with np.load(os.path.join(pred_dir, f)) as z:
            n = int(z["ts"].shape[0])
            for k in z.files:
                a = z[k]
                if a.shape == ():
                    a = np.repeat(a, n)
                if a.shape[0] != n:
                    continue
                cols.setdefault(k, []).append(a)
    table = {k: np.concatenate(v) for k, v in cols.items()}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        out = out_path or os.path.join(run_dir, "predictions.parquet")
        pq.write_table(pa.table(table), out, compression="zstd")
        return out
    except Exception:
        import gzip
        out = out_path or os.path.join(run_dir, "predictions.csv.gz")
        keys = list(table.keys())
        with gzip.open(out, "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for i in range(len(table[keys[0]])):
                w.writerow([table[k][i] for k in keys])
        return out
