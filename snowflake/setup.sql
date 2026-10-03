/* =====================================================================
   setup.sql -- create everything the Bitcoin price predictor needs
                inside Snowflake.

   Run top to bottom in a worksheet as a role that can create databases
   and compute pools (ACCOUNTADMIN, or a role with the right grants).

   The notebook itself runs with NO internet access: the dataset is staged
   from the repository once, and training reads it from local disk.
   ===================================================================== */

-- ---------------------------------------------------------------------
-- 1. Database, schema, warehouse
-- ---------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS BTCPRED;
CREATE SCHEMA   IF NOT EXISTS BTCPRED.CORE;
USE SCHEMA BTCPRED.CORE;

CREATE WAREHOUSE IF NOT EXISTS BTCPRED_WH
  WAREHOUSE_SIZE = 'MEDIUM'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

-- ---------------------------------------------------------------------
-- 2. Stages: one for the code, one for the ~2.5 GB dataset
-- ---------------------------------------------------------------------
CREATE STAGE IF NOT EXISTS BTCPRED_CODE
  DIRECTORY = (ENABLE = TRUE)
  COMMENT = 'src/, scripts/, train.py -- the predictor itself';

CREATE STAGE IF NOT EXISTS BTCPRED_DATA
  DIRECTORY = (ENABLE = TRUE)
  COMMENT = 'BTCUSDT 1-second compact chunks (.npz) + MANIFEST.json';

/* ---------------------------------------------------------------------
   3. Upload from your laptop with SnowSQL / the Snowflake CLI.
      (PUT only works from a client, not from a worksheet.)

     snow sql -q "PUT file://./src/btcpred/*.py       @BTCPRED.CORE.BTCPRED_CODE/btcpred/ OVERWRITE=TRUE AUTO_COMPRESS=FALSE"
     snow sql -q "PUT file://./train.py               @BTCPRED.CORE.BTCPRED_CODE/        OVERWRITE=TRUE AUTO_COMPRESS=FALSE"
     snow sql -q "PUT file://./data/btcusdt_1s/*.npz  @BTCPRED.CORE.BTCPRED_DATA/        OVERWRITE=TRUE AUTO_COMPRESS=FALSE PARALLEL=8"
     snow sql -q "PUT file://./data/btcusdt_1s/MANIFEST.json @BTCPRED.CORE.BTCPRED_DATA/ OVERWRITE=TRUE AUTO_COMPRESS=FALSE"

   Alternatively, if your account HAS outbound network access, attach the
   git repository directly (skip the PUTs entirely):

     CREATE OR REPLACE API INTEGRATION GH_API
       API_PROVIDER = GIT_HTTPS_API
       API_ALLOWED_PREFIXES = ('https://github.com/')
       ENABLED = TRUE;
     CREATE OR REPLACE GIT REPOSITORY BTCPRED_REPO
       API_INTEGRATION = GH_API
       ORIGIN = 'https://github.com/<owner>/pridictior_snow4.git';
     ALTER GIT REPOSITORY BTCPRED_REPO FETCH;
   --------------------------------------------------------------------- */

LIST @BTCPRED_DATA;   -- expect ~110 .npz chunks + MANIFEST.json

-- ---------------------------------------------------------------------
-- 4. Result tables (the notebook writes into these)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS BTC_PREDICTIONS (
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
);

CREATE TABLE IF NOT EXISTS BTC_TRAIN_METRICS (
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
);

-- ---------------------------------------------------------------------
-- 5. GPU compute pool (optional -- Container Runtime notebooks only).
--    With a GPU the notebook automatically switches to the intra-GPU
--    distributed path: several DDP ranks sharing one card.
-- ---------------------------------------------------------------------
CREATE COMPUTE POOL IF NOT EXISTS BTCPRED_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_S          -- 1x NVIDIA A10G, 24 GB
  AUTO_RESUME = TRUE
  AUTO_SUSPEND_SECS = 600;

-- CPU-only alternative:
-- CREATE COMPUTE POOL IF NOT EXISTS BTCPRED_CPU_POOL
--   MIN_NODES = 1 MAX_NODES = 1 INSTANCE_FAMILY = CPU_X64_M;

-- ---------------------------------------------------------------------
-- 6. Create the notebook from the staged file
-- ---------------------------------------------------------------------
-- Container Runtime (GPU):
CREATE OR REPLACE NOTEBOOK BTCPRED_LIVE
  FROM '@BTCPRED.CORE.BTCPRED_CODE/'
  MAIN_FILE = 'snowflake_live_training.ipynb'
  QUERY_WAREHOUSE = BTCPRED_WH
  RUNTIME_NAME = 'SYSTEM$GPU_RUNTIME'
  COMPUTE_POOL = BTCPRED_GPU_POOL;

-- Warehouse runtime (no container, numpy backend):
-- CREATE OR REPLACE NOTEBOOK BTCPRED_LIVE
--   FROM '@BTCPRED.CORE.BTCPRED_CODE/'
--   MAIN_FILE = 'snowflake_live_training.ipynb'
--   QUERY_WAREHOUSE = BTCPRED_WH;

ALTER NOTEBOOK BTCPRED_LIVE ADD LIVE VERSION FROM LAST;

-- ---------------------------------------------------------------------
-- 7. Useful queries once a run has written its results
-- ---------------------------------------------------------------------
-- headline accuracy over the last 100 windows
SELECT run_id,
       AVG(band_accuracy)  AS band_10bps,
       AVG(direction_acc)  AS direction,
       AVG(skill_vs_naive) AS skill,
       AVG(jitter_permille) AS jitter,
       COUNT(*)            AS windows
FROM   BTC_TRAIN_METRICS
QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY window DESC) <= 100
GROUP BY run_id;

-- actual vs predicted, ready to drop into a Snowsight line chart
SELECT ts,
       price_now,
       pred_price,
       true_future_price,
       ABS(pred_price - true_future_price) / true_future_price * 10000 AS err_bps
FROM   BTC_PREDICTIONS
WHERE  run_id = (SELECT MAX(run_id) FROM BTC_PREDICTIONS)
ORDER  BY ts
LIMIT  20000;
