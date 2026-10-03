"""
snowflake_io.py -- optional bridge between the compact dataset / run store and
Snowflake objects.

Nothing in this module is required for training.  The training path reads the
``.npz`` chunks straight off the filesystem, which is what makes the whole
thing work with **no internet and no Snowflake connection at all**.

What this adds, when a session happens to be available:

* ``stage_to_local()``   pull the chunk files from an internal stage into a
                         local directory once, at session start, so training
                         reads local disk afterwards;
* ``local_to_stage()``   push a freshly built dataset up to a stage;
* ``write_predictions()``write the per-second prediction tape into a table so
                         it can be queried with SQL / fed to a dashboard;
* ``write_metrics()``    same for the per-window metrics.

Every function degrades to a clear error message if ``snowflake.snowpark`` is
not importable, rather than breaking the import of the package.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

import numpy as np


def get_session(session: Optional[Any] = None):
    """Return the active Snowpark session (inside a Snowflake notebook this
    already exists), or raise with an actionable message."""
    if session is not None:
        return session
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception as e:                            # noqa: BLE001
        raise RuntimeError(
            "No active Snowpark session. Inside a Snowflake notebook one "
            "exists automatically; outside, pass session=Session.builder..."
        ) from e


# --------------------------------------------------------------------------
def stage_to_local(stage: str, local_dir: str, session: Optional[Any] = None,
                   pattern: str = "*") -> int:
    """
    Download dataset chunks from an internal stage to ``local_dir``.

    Call this once in the first notebook cell; everything afterwards is plain
    local file I/O, so the training loop never touches the network.
    """
    s = get_session(session)
    os.makedirs(local_dir, exist_ok=True)
    stage = stage.rstrip("/")
    rows = s.sql(f"LIST {stage}").collect()
    got = 0
    for r in rows:
        name = r["name"] if "name" in r.as_dict() else list(r.as_dict().values())[0]
        base = os.path.basename(name)
        if pattern != "*" and pattern not in base:
            continue
        if os.path.exists(os.path.join(local_dir, base)):
            got += 1
            continue
        s.file.get(f"@{name}", local_dir)
        got += 1
    return got


def local_to_stage(local_dir: str, stage: str, session: Optional[Any] = None,
                   pattern: str = "*.npz") -> int:
    """Upload chunk files (and the manifest) to an internal stage."""
    s = get_session(session)
    stage = stage.rstrip("/")
    files = sorted(glob.glob(os.path.join(local_dir, pattern)))
    man = os.path.join(local_dir, "MANIFEST.json")
    if os.path.exists(man):
        files.append(man)
    for f in files:
        s.file.put(f, stage, auto_compress=False, overwrite=True)
    return len(files)


# --------------------------------------------------------------------------
PRED_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
  run_id            STRING,
  window_id         BIGINT,
  rank              INT,
  ts                TIMESTAMP_NTZ,
  price_now         DOUBLE,
  mu_raw            DOUBLE,
  mu_smooth         DOUBLE,
  sigma             DOUBLE,
  pred_price        DOUBLE,
  true_future_price DOUBLE,
  target_permille   DOUBLE
)
"""

METRIC_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
  run_id          STRING,
  window          BIGINT,
  utc             STRING,
  loss            DOUBLE,
  mae_permille    DOUBLE,
  band_accuracy   DOUBLE,
  direction_acc   DOUBLE,
  skill_vs_naive  DOUBLE,
  calibration     DOUBLE,
  jitter_permille DOUBLE,
  sim_sec_per_sec DOUBLE
)
"""


def write_predictions(run_dir: str, table: str = "BTC_PREDICTIONS",
                      session: Optional[Any] = None,
                      run_id: Optional[str] = None,
                      max_windows: Optional[int] = None) -> int:
    """Append every stored per-second prediction into a Snowflake table."""
    import pandas as pd
    s = get_session(session)
    s.sql(PRED_DDL.format(table=table)).collect()
    rid = run_id or os.path.basename(os.path.abspath(run_dir))
    files = sorted(glob.glob(os.path.join(run_dir, "predictions", "*.npz")))
    if max_windows:
        files = files[-max_windows:]
    total = 0
    for f in files:
        with np.load(f) as z:
            n = int(z["ts"].shape[0])
            df = pd.DataFrame({
                "RUN_ID": rid,
                "WINDOW_ID": int(z["window"]),
                "RANK": int(z["rank"]),
                "TS": pd.to_datetime(z["ts"], unit="s"),
                "PRICE_NOW": z["price_now"].astype("float64"),
                "MU_RAW": z["mu_raw"].astype("float64"),
                "MU_SMOOTH": z["mu_smooth"].astype("float64"),
                "SIGMA": z["sigma"].astype("float64"),
                "PRED_PRICE": z["pred_price"].astype("float64"),
                "TRUE_FUTURE_PRICE": (z["true_future_price"].astype("float64")
                                      if "true_future_price" in z.files
                                      else np.full(n, np.nan)),
                "TARGET_PERMILLE": (z["target_permille"].astype("float64")
                                    if "target_permille" in z.files
                                    else np.full(n, np.nan)),
            })
        s.create_dataframe(df).write.mode("append").save_as_table(table)
        total += len(df)
    return total


def write_metrics(run_dir: str, table: str = "BTC_TRAIN_METRICS",
                  session: Optional[Any] = None,
                  run_id: Optional[str] = None) -> int:
    import csv
    import pandas as pd
    s = get_session(session)
    s.sql(METRIC_DDL.format(table=table)).collect()
    rid = run_id or os.path.basename(os.path.abspath(run_dir))
    p = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(p):
        return 0
    with open(p) as fh:
        rows: List[Dict[str, str]] = list(csv.DictReader(fh))
    keep = ["window", "utc", "loss", "mae_permille", "band_accuracy",
            "direction_acc", "skill_vs_naive", "calibration",
            "jitter_permille", "sim_sec_per_sec"]
    out = []
    for r in rows:
        d: Dict[str, Any] = {"RUN_ID": rid}
        for k in keep:
            v = r.get(k, "")
            if k == "utc":
                d["UTC"] = v
            else:
                try:
                    d[k.upper()] = float(v) if v != "" else None
                except ValueError:
                    d[k.upper()] = None
        out.append(d)
    s.create_dataframe(pd.DataFrame(out)).write.mode("append").save_as_table(table)
    return len(out)
