"""
btcpred -- second-by-second Bitcoin price prediction.

A live market simulator replays the real 1-second tape, a CPU Market Analyser
turns it into causal signals, and a GPU Price Predictor forecasts the price
30 minutes ahead, refreshing its forecast every second.

Everything runs offline: the dataset ships with the repository in a compact
NumPy-only codec, so there is no network call anywhere in the training path.

Modules
-------
compact       storage codec + chunk index for the 288 M-bar dataset
simulator     live market simulator (tick stream and 30-minute window stream)
analyser      Market Analyser -- 40 causal signals, O(1) per second (CPU)
model         Price Predictor -- causal TCN -> GRU -> (mu, sigma, direction),
              with a hand-differentiated NumPy fallback
stability     causal smoothing so the forecast is tradeable
metrics       accuracy / skill / calibration / stability accounting
storage       per-second prediction tape and per-window checkpoints
distributed   intra-GPU DDP (several ranks sharing one card) and CPU fallback
trainer       the 30-minute-epoch loop with delayed reward/loss
viz           animation, live notebook chart, self-contained HTML replay
env           hardware detection and automatic model sizing
snowflake_io  optional stage/table bridge (never required for training)

Quick start
-----------
>>> from btcpred.compact import ChunkIndex
>>> from btcpred.trainer import TrainConfig, train
>>> print(ChunkIndex("data/btcusdt_1s").describe())
>>> train(TrainConfig(max_windows=100))
"""

__version__ = "1.0.0"
__all__ = [
    "compact", "simulator", "analyser", "model", "stability", "metrics",
    "storage", "distributed", "trainer", "viz", "env", "snowflake_io",
]

HORIZON_SECONDS = 1800      # how far ahead the model predicts
EPOCH_SECONDS = 1800        # one training epoch of simulated market time
