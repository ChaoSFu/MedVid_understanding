#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_sglang_trainval_model.sh \
    --run-name qwen35_4b \
    --model-path /path/to/model \
    --gpu 0 \
    --port 30002

Optional:
  --tp 1
  --concurrency 4
  --max-completion-tokens 1024
  --data-path data_json/init_datas/medvidu_eccv2026_trainval.json
  --old-data-root /root/data
  --new-data-root /mnt/hdd3/huihui/hh_datas/MedVidU/valdata
  --media-schema image_sequence
  --mem-fraction-static 0.80
  --host 0.0.0.0
USAGE
}

RUN_NAME=""
MODEL_PATH=""
GPU=""
PORT=""
TP="1"
CONCURRENCY="4"
MAX_COMPLETION_TOKENS="1024"
DATA_PATH="data_json/init_datas/medvidu_eccv2026_trainval.json"
OLD_DATA_ROOT="/root/data"
NEW_DATA_ROOT="/mnt/hdd3/huihui/hh_datas/MedVidU/valdata"
MEDIA_SCHEMA="image_sequence"
MEM_FRACTION_STATIC="0.80"
HOST="0.0.0.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --tp)
      TP="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --max-completion-tokens)
      MAX_COMPLETION_TOKENS="$2"
      shift 2
      ;;
    --data-path)
      DATA_PATH="$2"
      shift 2
      ;;
    --old-data-root)
      OLD_DATA_ROOT="$2"
      shift 2
      ;;
    --new-data-root)
      NEW_DATA_ROOT="$2"
      shift 2
      ;;
    --media-schema)
      MEDIA_SCHEMA="$2"
      shift 2
      ;;
    --mem-fraction-static)
      MEM_FRACTION_STATIC="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_NAME" || -z "$MODEL_PATH" || -z "$GPU" || -z "$PORT" ]]; then
  echo "Missing required --run-name, --model-path, --gpu, or --port." >&2
  usage >&2
  exit 2
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path does not exist: $MODEL_PATH" >&2
  exit 2
fi

mkdir -p "logs/sglang" "results/${RUN_NAME}"

SERVER_LOG="logs/sglang/server_${RUN_NAME}_gpu${GPU}_port${PORT}.log"
INFER_LOG="logs/sglang/infer_${RUN_NAME}.log"
BASE_URL="http://127.0.0.1:${PORT}/v1"

echo "Starting SGLang server for ${RUN_NAME}"
echo "  model: ${MODEL_PATH}"
echo "  gpu: ${GPU}"
echo "  port: ${PORT}"
echo "  server log: ${SERVER_LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" nohup python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp "${TP}" \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --watchdog-timeout 1200 \
  > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
echo "${SERVER_PID}" > "logs/sglang/server_${RUN_NAME}.pid"

echo "Waiting for SGLang health endpoint..."
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "SGLang is ready."
    break
  fi

  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "SGLang server exited early. See ${SERVER_LOG}" >&2
    exit 1
  fi

  sleep 5
done

if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "SGLang did not become ready within 10 minutes. See ${SERVER_LOG}" >&2
  exit 1
fi

echo "Starting async trainval inference for ${RUN_NAME}"
echo "  base url: ${BASE_URL}"
echo "  inference log: ${INFER_LOG}"

nohup python3 inference/qwen38_27b_sglang_async_api_infer.py \
  --data_path "${DATA_PATH}" \
  --old_data_root "${OLD_DATA_ROOT}" \
  --new_data_root "${NEW_DATA_ROOT}" \
  --base_url "${BASE_URL}" \
  --model "${MODEL_PATH}" \
  --qa_types all \
  --concurrency "${CONCURRENCY}" \
  --media_schema "${MEDIA_SCHEMA}" \
  --max_completion_tokens "${MAX_COMPLETION_TOKENS}" \
  --output_path "results/${RUN_NAME}/results.json" \
  --submission_path "results/${RUN_NAME}/submission.json" \
  --failure_path "results/${RUN_NAME}/failures.json" \
  --log_path "results/${RUN_NAME}/inference.log" \
  > "${INFER_LOG}" 2>&1 &

INFER_PID=$!
echo "${INFER_PID}" > "logs/sglang/infer_${RUN_NAME}.pid"

echo "Launched."
echo "  server pid: ${SERVER_PID}"
echo "  infer pid: ${INFER_PID}"
echo "  tail server: tail -f ${SERVER_LOG}"
echo "  tail infer:  tail -f ${INFER_LOG}"
