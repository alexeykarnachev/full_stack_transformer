#!/bin/sh

PORT=$1
WORKERS=$2
LOGS_DIR=$3
CHECKPOINT_PATH=$4

TIMEOUT=600
WORKER_CLASS="uvicorn.workers.UvicornWorker"
MODULE_PATH="lm_trainer.scripts.run_application"
ACCESS_LOGFILE=$LOGS_DIR"/gunicorn_access.log"
ERROR_LOGFILE=$LOGS_DIR"/gunicorn_error.log"
LOG_LEVEL="DEBUG"

exec gunicorn ${MODULE_PATH}":prepare(checkpoint_path=${CHECKPOINT_PATH})" \
-b :"${PORT}" \
--timeout "${TIMEOUT}" \
-k "${WORKER_CLASS}" \
--workers "${WORKERS}" \
--access-logfile "${ACCESS_LOGFILE}" \
--error-logfile "${ERROR_LOGFILE}" \
--log-level "${LOG_LEVEL}" \
