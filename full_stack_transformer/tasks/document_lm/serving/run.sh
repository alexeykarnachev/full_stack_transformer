#!/bin/sh

PORT=$1
WORKERS=$2
LOGS_DIR=$3
CHECKPOINT_PATH=$4
DEVICE=$5

TIMEOUT=600
WORKER_CLASS="uvicorn.workers.UvicornWorker"
MODULE_PATH="full_stack_transformer.tasks.document_lm.serving.app"
ACCESS_LOGFILE=$LOGS_DIR"/gunicorn_access.log"
ERROR_LOGFILE=$LOGS_DIR"/gunicorn_error.log"
LOG_LEVEL="DEBUG"

mkdir -p "${LOGS_DIR}"

exec gunicorn ${MODULE_PATH}":prepare(checkpoint_path='${CHECKPOINT_PATH}', device='${DEVICE}', logs_dir='${LOGS_DIR}')" \
-b :"${PORT}" \
--timeout "${TIMEOUT}" \
-k "${WORKER_CLASS}" \
--workers "${WORKERS}" \
--access-logfile "${ACCESS_LOGFILE}" \
--error-logfile "${ERROR_LOGFILE}" \
--log-level "${LOG_LEVEL}" \
