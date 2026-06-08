#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  run_eval.sh infer --config CONFIG --input INPUT --output-dir OUTDIR [options]
  run_eval.sh infer --config CONFIG --input-list INPUT_LIST --output-dir OUTDIR [options]
  run_eval.sh stats [--config CONFIG] --input PREDICTIONS --output-dir OUTDIR [options]
  run_eval.sh stats [--config CONFIG] --input-list PREDICTION_LIST --output-dir OUTDIR [options]
  run_eval.sh both  --config CONFIG --input INPUT --output-dir OUTDIR [options]
  run_eval.sh both  --config CONFIG --input-list INPUT_LIST --output-dir OUTDIR [options]

Modes:
  infer    Run model inference only.
  stats    Compute WER/CER/SER statistics only.
  both     Run inference, then statistics.

Common options:
  --config PATH              YAML config.
  --env-dir PATH             Python environment directory containing bin/activate.
  --input PATH               Input JSON/JSONL. In stats mode this is the predictions file.
  --input-list PATH          JSON object mapping test set names to files. Alias: --input_list.
  --output-dir PATH          Directory for outputs.
  --prediction-field NAME    Prediction field name. Default: prediction.
  --limit N                  Inference-only record limit.

Stats options:
  --predictions PATH         Prediction file for stats mode or both mode override.
  --inference-summary PATH   inference_summary.json path.
  --inference-seconds SEC    Override inference seconds.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
  usage
  exit 0
fi

MODE="$1"
shift

CONFIG=""
ENV_DIR=""
INPUT=""
INPUT_LIST=""
OUTPUT_DIR=""
PREDICTIONS=""
PREDICTION_FIELD=""
LIMIT=""
INFERENCE_SUMMARY=""
INFERENCE_SECONDS=""
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="${2:?missing value for --config}"
      shift 2
      ;;
    --env-dir|--env-path)
      ENV_DIR="${2:?missing value for $1}"
      shift 2
      ;;
    --input)
      INPUT="${2:?missing value for --input}"
      shift 2
      ;;
    --input-list|--input_list)
      INPUT_LIST="${2:?missing value for $1}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --predictions)
      PREDICTIONS="${2:?missing value for --predictions}"
      shift 2
      ;;
    --prediction-field)
      PREDICTION_FIELD="${2:?missing value for --prediction-field}"
      shift 2
      ;;
    --limit)
      LIMIT="${2:?missing value for --limit}"
      shift 2
      ;;
    --inference-summary)
      INFERENCE_SUMMARY="${2:?missing value for --inference-summary}"
      shift 2
      ;;
    --inference-seconds)
      INFERENCE_SECONDS="${2:?missing value for --inference-seconds}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "${OUTPUT_DIR}" ]; then
  echo "--output-dir is required" >&2
  usage >&2
  exit 2
fi
if [ -z "${INPUT}" ] && [ -z "${INPUT_LIST}" ]; then
  echo "one of --input or --input-list is required" >&2
  usage >&2
  exit 2
fi
if [ -n "${INPUT}" ] && [ -n "${INPUT_LIST}" ]; then
  echo "use only one of --input or --input-list" >&2
  usage >&2
  exit 2
fi
if [ "${MODE}" != "stats" ] && [ -z "${CONFIG}" ]; then
  echo "--config is required for ${MODE} mode" >&2
  usage >&2
  exit 2
fi
if [ -n "${ENV_DIR}" ]; then
  ENV_DIR="${ENV_DIR%/}"
  ACTIVATE_SCRIPT="${ENV_DIR}/bin/activate"
  if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
    echo "--env-dir must contain bin/activate: ${ACTIVATE_SCRIPT}" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${ACTIVATE_SCRIPT}"
  PYTHON_BIN="${ENV_DIR}/bin/python"
  if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python)"
  fi
fi

mkdir -p "${OUTPUT_DIR}"

infer_args=(--config "${CONFIG}" --output-dir "${OUTPUT_DIR}")
if [ -n "${INPUT_LIST}" ]; then
  infer_args+=(--input-list "${INPUT_LIST}")
else
  infer_args+=(--input "${INPUT}")
fi
stats_args=(--output-dir "${OUTPUT_DIR}")
if [ -n "${CONFIG}" ]; then
  stats_args+=(--config "${CONFIG}")
fi

if [ -n "${PREDICTION_FIELD}" ]; then
  infer_args+=(--prediction-field "${PREDICTION_FIELD}")
  stats_args+=(--prediction-field "${PREDICTION_FIELD}")
fi
if [ -n "${LIMIT}" ]; then
  infer_args+=(--limit "${LIMIT}")
fi
if [ -n "${INFERENCE_SUMMARY}" ]; then
  stats_args+=(--inference-summary "${INFERENCE_SUMMARY}")
fi
if [ -n "${INFERENCE_SECONDS}" ]; then
  stats_args+=(--inference-seconds "${INFERENCE_SECONDS}")
fi

case "${MODE}" in
  infer)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_inference.py" "${infer_args[@]}"
    ;;
  stats)
    if [ -n "${INPUT_LIST}" ]; then
      "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_stats.py" "${stats_args[@]}" --input-list "${INPUT_LIST}"
    else
      stats_input="${PREDICTIONS:-${INPUT}}"
      "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_stats.py" "${stats_args[@]}" --input "${stats_input}"
    fi
    ;;
  both)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_inference.py" "${infer_args[@]}"
    if [ -n "${INPUT_LIST}" ]; then
      stats_input_list="${PREDICTIONS:-${OUTPUT_DIR}/prediction_list.json}"
      if [ -z "${INFERENCE_SUMMARY}" ]; then
        stats_args+=(--inference-summary "${OUTPUT_DIR}/batch_inference_summary.json")
      fi
      "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_stats.py" "${stats_args[@]}" --input-list "${stats_input_list}"
    else
      if [ -z "${PREDICTIONS}" ]; then
        case "${INPUT}" in
          *.jsonl) PREDICTIONS="${OUTPUT_DIR}/predictions.jsonl" ;;
          *) PREDICTIONS="${OUTPUT_DIR}/predictions.json" ;;
        esac
      fi
      if [ -z "${INFERENCE_SUMMARY}" ]; then
        stats_args+=(--inference-summary "${OUTPUT_DIR}/inference_summary.json")
      fi
      "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_stats.py" "${stats_args[@]}" --input "${PREDICTIONS}"
    fi
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
