#!/usr/bin/env bash
#
# Run NeMo simulstream inference on longform audio.
# Input: either a NeMo manifest (JSONL) or a text file listing wav paths (one per line).
#
# Usage with manifest:
#   ./run_simulstream_inference.sh \
#     manifest=/path/to/longform_manifest.jsonl \
#     output-dir=/path/to/output_base \
#     src-lang=en tgt-lang=ru \
#     nemo-config=examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml \
#     [llm-model="Qwen/Qwen2.5-7B-Instruct"]
#
# Usage with wav list:
#   ./run_simulstream_inference.sh \
#     wav-list=/path/to/audio_list.txt \
#     output-dir=/path/to/output_base \
#     src-lang=en tgt-lang=ru \
#     nemo-config=... \
#     [llm-model="Qwen/Qwen2.5-7B-Instruct"]
#
# Output: OUTPUT_DIR (output-dir/input_name/config_name/llm_model) with simulstream_output.json
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$NEMO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/home/lgrigoryan/data/hf_cache}"
SCRIPT_START_TIME="$(date +%Y%m%d_%H%M%S)"
export TORCH_HOME="${TORCH_HOME:-/home/lgrigoryan/data/torch_cache/$SCRIPT_START_TIME}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/lgrigoryan/data/triton_cache/$SCRIPT_START_TIME}"
export TORCH_EXTENSIONS_DIR=/lustre/fsw/portfolios/convai/users/lgrigoryan/torch_ext
export MEGATRON_DISABLE_UNIFIED_MEMORY=1
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR"

MANIFEST=""
WAV_LIST=""

OUTPUT_DIR_BASE=""

SRC_LANG="en"
TGT_LANG=""
NEMO_CONFIG=""

LLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
EUROLLM_MAX_SEQ_LENGTH="${EUROLLM_MAX_SEQ_LENGTH:-4096}"

CACHE_ATT_CONTEXT_SIZE="13"
BUFFERED_CHUNK_SIZE="1.12"
BUFFERED_LEFT_PADDING_SIZE="5.6"
BUFFERED_RIGHT_PADDING_SIZE="0.56"

FORCE="false"

usage() {
  echo "Usage: $0 (manifest=PATH | wav-list=PATH) output-dir=DIR src-lang=LANG tgt-lang=LANG nemo-config=YAML [OPTIONS]"
  echo ""
  echo "Input (one required):"
  echo "  manifest=PATH   NeMo manifest JSONL (audio_filepath per line)"
  echo "  wav-list=PATH   Text file with one wav path per line"
  echo ""
  echo "Required:"
  echo "  output-dir=DIR   Base directory for outputs (subdir will be created)"
  echo "  src-lang=LANG    Source language code (e.g. en, ru)"
  echo "  tgt-lang=LANG    Target language code (e.g. ru, en)"
  echo "  nemo-config=YAML NeMo streaming config (e.g. cache_aware_rnnt.yaml)"
  echo ""
  echo "Optional:"
  echo "  llm-model=MODEL  LLM model (default: Qwen/Qwen2.5-7B-Instruct)"
  echo "  force=true|false Re-run and overwrite existing simulstream_output.json (default: false)"
  echo "  cache-att-context-size=INT   Required for cache_aware_rnnt naming/override (e.g. 13)"
  echo "  buffered-chunk-size=FLOAT    Required for buffered_rnnt naming/override"
  echo "  buffered-left-padding-size=FLOAT  Required for buffered_rnnt naming/override"
  echo "  buffered-right-padding-size=FLOAT Required for buffered_rnnt naming/override"
  exit 1
}

for arg in "$@"; do
  case "$arg" in
    manifest=*)      MANIFEST="${arg#*=}" ;;
    wav-list=*)      WAV_LIST="${arg#*=}" ;;
    output-dir=*)    OUTPUT_DIR_BASE="${arg#*=}" ;;
    src-lang=*)      SRC_LANG="${arg#*=}" ;;
    tgt-lang=*)      TGT_LANG="${arg#*=}" ;;
    nemo-config=*)   NEMO_CONFIG="${arg#*=}" ;;
    llm-model=*)     LLM_MODEL="${arg#*=}" ;;
    cache-att-context-size=*) CACHE_ATT_CONTEXT_SIZE="${arg#*=}" ;;
    buffered-chunk-size=*) BUFFERED_CHUNK_SIZE="${arg#*=}" ;;
    buffered-left-padding-size=*) BUFFERED_LEFT_PADDING_SIZE="${arg#*=}" ;;
    buffered-right-padding-size=*) BUFFERED_RIGHT_PADDING_SIZE="${arg#*=}" ;;
    force=*)
      FORCE_VALUE="${arg#*=}"
      case "${FORCE_VALUE,,}" in
        1|true|yes|on) FORCE="true" ;;
        0|false|no|off|"") FORCE="false" ;;
        *) echo "Error: invalid force value '$FORCE_VALUE' (use true/false)"; usage ;;
      esac
      ;;
    -h|--help|help=true) usage ;;
    *=*)             echo "Unknown option: $arg"; usage ;;
    *)               echo "Invalid argument format (expected key=value): $arg"; usage ;;
  esac
done

[[ -z "$OUTPUT_DIR_BASE" ]] && echo "Error: missing required argument: output-dir=DIR" && usage
[[ -z "$SRC_LANG" ]] && echo "Error: missing required argument: src-lang=LANG" && usage
[[ -z "$TGT_LANG" ]] && echo "Error: missing required argument: tgt-lang=LANG" && usage
[[ -z "$NEMO_CONFIG" ]] && echo "Error: missing required argument: nemo-config=YAML" && usage

if [[ -n "$MANIFEST" && -n "$WAV_LIST" ]]; then
  echo "Error: use either manifest=PATH or wav-list=PATH, not both."
  usage
fi
if [[ -z "$MANIFEST" && -z "$WAV_LIST" ]]; then
  echo "Error: provide either manifest=PATH or wav-list=PATH."
  usage
fi

if [[ -n "$MANIFEST" ]]; then
  INPUT_PATH="$MANIFEST"
  if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: manifest not found: $MANIFEST"
    exit 1
  fi
else
  INPUT_PATH="$WAV_LIST"
  if [[ ! -f "$WAV_LIST" ]]; then
    echo "Error: wav-list not found: $WAV_LIST"
    exit 1
  fi
fi

if [[ ! -f "$NEMO_CONFIG" ]] && [[ -f "$NEMO_ROOT/$NEMO_CONFIG" ]]; then
  NEMO_CONFIG="$NEMO_ROOT/$NEMO_CONFIG"
fi
if [[ ! -f "$NEMO_CONFIG" ]]; then
  echo "Error: nemo config not found: $NEMO_CONFIG"
  exit 1
fi
NEMO_CONFIG_ABS="$(realpath "$NEMO_CONFIG")"

# Output subdir name from input file name
INPUT_NAME=$(basename "$INPUT_PATH")
INPUT_NAME="${INPUT_NAME%.jsonl}"
INPUT_NAME="${INPUT_NAME%.json}"
INPUT_NAME="${INPUT_NAME%.txt}"
INPUT_NAME=${INPUT_NAME#manifest_}

CONFIG_NAME=$(basename "$NEMO_CONFIG" .yaml)
LLM_MODEL_SAFE=${LLM_MODEL//\//_}
OUTPUT_DIR="$OUTPUT_DIR_BASE/${INPUT_NAME}/${CONFIG_NAME}/${LLM_MODEL_SAFE}"
EXTRA_OVERRIDES=()

if [[ "$CONFIG_NAME" == "cache_aware_rnnt" ]]; then
  if [[ -z "$CACHE_ATT_CONTEXT_SIZE" ]]; then
    echo "Error: cache-att-context-size=INT is required for cache_aware_rnnt."
    exit 1
  fi
  OUTPUT_DIR="$OUTPUT_DIR_BASE/${INPUT_NAME}/${CONFIG_NAME}_${CACHE_ATT_CONTEXT_SIZE}/${LLM_MODEL_SAFE}"
  EXTRA_OVERRIDES+=("streaming.att_context_size=[70,${CACHE_ATT_CONTEXT_SIZE}]")
fi

if [[ "$CONFIG_NAME" == "buffered_rnnt" ]]; then
  if [[ -z "$BUFFERED_CHUNK_SIZE" || -z "$BUFFERED_LEFT_PADDING_SIZE" || -z "$BUFFERED_RIGHT_PADDING_SIZE" ]]; then
    echo "Error: buffered-chunk-size, buffered-left-padding-size, and buffered-right-padding-size are required for buffered_rnnt."
    exit 1
  fi
  OUTPUT_DIR="$OUTPUT_DIR_BASE/${INPUT_NAME}/${CONFIG_NAME}_c${BUFFERED_CHUNK_SIZE}_l${BUFFERED_LEFT_PADDING_SIZE}_r${BUFFERED_RIGHT_PADDING_SIZE}/${LLM_MODEL_SAFE}"
  EXTRA_OVERRIDES+=(
    "streaming.chunk_size=${BUFFERED_CHUNK_SIZE}"
    "streaming.left_padding_size=${BUFFERED_LEFT_PADDING_SIZE}"
    "streaming.right_padding_size=${BUFFERED_RIGHT_PADDING_SIZE}"
  )
fi

if [[ "${LLM_MODEL,,}" == *"eurollm"* ]]; then
  EXTRA_OVERRIDES+=("nmt.llm_params.max_model_len=${EUROLLM_MAX_SEQ_LENGTH}")
fi

mkdir -p "$OUTPUT_DIR"
cd "$NEMO_ROOT"

INPUT_ABS="$(realpath "$INPUT_PATH")"
OUTPUT_DIR_ABS="$(realpath "$OUTPUT_DIR")"
HYPOTHESIS_JSON="$OUTPUT_DIR_ABS/simulstream_output.json"
INFERENCE_DONE_MARKER="$OUTPUT_DIR_ABS/.simulstream_inference_done"

echo "========== Run NeMo simulstream =========="
if [[ -f "$HYPOTHESIS_JSON" ]]; then
  if [[ ! -s "$HYPOTHESIS_JSON" ]]; then
    echo "Simulstream output exists but is empty: $HYPOTHESIS_JSON"
    echo "Removing empty output and re-running inference."
    rm -f "$HYPOTHESIS_JSON"
  elif [[ "$FORCE" == "true" ]]; then
    echo "Simulstream output already exists at: $HYPOTHESIS_JSON"
    echo "force=true specified, overwriting existing output."
    rm -f "$HYPOTHESIS_JSON"
    rm -f "$INFERENCE_DONE_MARKER"
  elif [[ ! -f "$INFERENCE_DONE_MARKER" ]]; then
    echo "Simulstream output exists but previous inference did not finish cleanly."
    echo "Removing stale output and re-running inference."
    rm -f "$HYPOTHESIS_JSON"
  else
    echo "Simulstream output already exists at: $HYPOTHESIS_JSON"
    echo "Skipping inference (set force=true to overwrite)."
    echo ""
    echo "Done. Output directory: $OUTPUT_DIR_ABS"
    exit 0
  fi
fi

rm -f "$INFERENCE_DONE_MARKER"
if [[ -n "$MANIFEST" ]]; then
  python nemo/collections/asr/inference/run_nemo_simulstream.py \
    --config "$NEMO_CONFIG_ABS" \
    --manifest "$INPUT_ABS" \
    --src-lang "$SRC_LANG" \
    --tgt-lang "$TGT_LANG" \
    --metrics-log "$HYPOTHESIS_JSON" \
    "nmt.model_name=$LLM_MODEL" \
    "${EXTRA_OVERRIDES[@]}"
else
  python nemo/collections/asr/inference/run_nemo_simulstream.py \
    --config "$NEMO_CONFIG_ABS" \
    --wav-list "$INPUT_ABS" \
    --src-lang "$SRC_LANG" \
    --tgt-lang "$TGT_LANG" \
    --metrics-log "$HYPOTHESIS_JSON" \
    "nmt.model_name=$LLM_MODEL" \
    "${EXTRA_OVERRIDES[@]}"
fi
touch "$INFERENCE_DONE_MARKER"
echo "Simulstream output written to: $HYPOTHESIS_JSON"

echo ""
echo "Done. Output directory: $OUTPUT_DIR_ABS"
