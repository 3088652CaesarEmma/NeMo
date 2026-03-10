#!/usr/bin/env bash
#
# Run simulstream inference from a manifest and optionally evaluate with omnisteval.
#
# 1. Converts manifest (can be segments manifest) to audio definitions (references.txt, transcripts.txt, audio_definitions.yaml).
# 2. Runs NeMo simulstream and writes hypothesis JSON to OUTPUT_DIR.
# 3. If --speech-segmentation and --simulstream-config are set, runs omnisteval longform.
#
# Usage:
#   ./scripts/simulstream/run_simulstream_eval.sh \
#     --manifest /path/to/manifest.jsonl \
#     --output-dir /path/to/output \
#     --src-lang en --tgt-lang ru \
#     --nemo-config examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml \
#     [--speech-segmentation /path/to/gold_segments.yaml] \
#     [--simulstream-config /path/to/simulstream/config/nemo_cascade.yaml] \
#     [--run-name my_run] \
#     [--comet]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Use this repo's NeMo (e.g. host-mounted in Docker) instead of any pip-installed nemo in the env
export PYTHONPATH="$NEMO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
MANIFEST=""
SEGMENTS_MANIFEST=""
OUTPUT_DIR_BASE=""
SRC_LANG=""
TGT_LANG=""
NEMO_CONFIG=""
SPEECH_SEGMENTATION=""
SIMULSTREAM_CONFIG=""
LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
COMET=""
HF_HOME=/home/lgrigoryan/data/hf_cache

usage() {
  echo "Usage: $0 --manifest PATH --output-dir DIR --src-lang LANG --tgt-lang LANG --nemo-config YAML [OPTIONS]"
  echo ""
  echo "Required:"
  echo "  --manifest PATH       Longform NeMo manifest JSONL (for inference)"
  echo "  --output-dir DIR      Base directory for outputs (subdir will be created)"
  echo "  --src-lang LANG       Source language code (e.g. en, ru)"
  echo "  --tgt-lang LANG       Target language code (e.g. ru, en)"
  echo "  --nemo-config YAML    NeMo streaming config (e.g. cache_aware_rnnt.yaml)"
  echo ""
  echo "Optional:"
  echo "  --segments-manifest PATH   Segments NeMo manifest JSONL (required for omnisteval evaluation)"
  echo "  --simulstream-config YAML    Simulstream config for omnisteval (e.g. nemo_cascade.yaml)"
  echo "  --llm-model MODEL            LLM model (default: Qwen/Qwen2.5-7B-Instruct)"
  echo "  --comet                     Pass --comet to omnisteval"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)            MANIFEST="$2"; shift 2 ;;
    --segments-manifest)   SEGMENTS_MANIFEST="$2"; shift 2 ;;
    --output-dir)          OUTPUT_DIR_BASE="$2"; shift 2 ;;
    --src-lang)            SRC_LANG="$2"; shift 2 ;;
    --tgt-lang)            TGT_LANG="$2"; shift 2 ;;
    --nemo-config)         NEMO_CONFIG="$2"; shift 2 ;;
    --simulstream-config)  SIMULSTREAM_CONFIG="$2"; shift 2 ;;
    --llm-model)           LLM_MODEL="$2"; shift 2 ;;
    --comet)               COMET="--comet"; shift ;;
    -h|--help)             usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

for var in MANIFEST OUTPUT_DIR_BASE SRC_LANG TGT_LANG NEMO_CONFIG; do
  if [[ -z "${!var}" ]]; then
    echo "Error: missing required argument: --${var,,}"
    usage
  fi
done

if [[ ! -f "$MANIFEST" ]]; then
  echo "Error: manifest not found: $MANIFEST"
  exit 1
fi
if [[ ! -f "$NEMO_CONFIG" ]]; then
  if [[ -f "$NEMO_ROOT/$NEMO_CONFIG" ]]; then
    NEMO_CONFIG="$NEMO_ROOT/$NEMO_CONFIG"
  else
    echo "Error: nemo config not found: $NEMO_CONFIG"
    exit 1
  fi
fi

# Determine subdirectory name based on config and manifest
CONFIG_NAME=$(basename "$NEMO_CONFIG" .yaml)
MANIFEST_NAME=$(basename "$MANIFEST" .jsonl)
# Remove 'manifest_' prefix if present for cleaner name
MANIFEST_NAME=${MANIFEST_NAME#manifest_}

# Sanitize LLM model name for directory usage (replace / with _)
LLM_MODEL_SAFE=${LLM_MODEL//\//_}
OUTPUT_DIR="$OUTPUT_DIR_BASE/${MANIFEST_NAME}/${CONFIG_NAME}/${LLM_MODEL_SAFE}"
mkdir -p "$OUTPUT_DIR"
cd "$NEMO_ROOT"

# Resolve paths so they are absolute for scripts that may change cwd
MANIFEST_ABS="$(realpath "$MANIFEST")"
SEGMENTS_MANIFEST_ABS=""
if [[ -n "$SEGMENTS_MANIFEST" ]]; then
  if [[ ! -f "$SEGMENTS_MANIFEST" ]]; then
      echo "Error: segments manifest not found: $SEGMENTS_MANIFEST"
      exit 1
  fi
  SEGMENTS_MANIFEST_ABS="$(realpath "$SEGMENTS_MANIFEST")"
fi

OUTPUT_DIR_ABS="$(realpath "$OUTPUT_DIR")"
NEMO_CONFIG_ABS="$(realpath "$NEMO_CONFIG")"

HYPOTHESIS_JSON="$OUTPUT_DIR_ABS/simulstream_output.json"

export HF_HOME=$HF_HOME

if [[ -n "$SEGMENTS_MANIFEST_ABS" ]]; then
  echo "========== 1. Create audio definitions from segments manifest =========="
  python "$SCRIPT_DIR/create_audio_definitions_from_manifest.py" \
    --manifest "$SEGMENTS_MANIFEST_ABS" \
    --output-dir "$OUTPUT_DIR_ABS"
else
  echo "Skipping audio definitions generation (no segments manifest provided)."
fi

echo ""
echo "========== 2. Run NeMo simulstream =========="
if [[ -f "$HYPOTHESIS_JSON" ]]; then
  echo "Simulstream output already exists at: $HYPOTHESIS_JSON"
  echo "Skipping inference."
else
  python nemo/collections/asr/inference/run_nemo_simulstream.py \
    --config "$NEMO_CONFIG_ABS" \
    --manifest "$MANIFEST_ABS" \
    --src-lang "$SRC_LANG" \
    --tgt-lang "$TGT_LANG" \
    --metrics-log "$HYPOTHESIS_JSON" \
    --nmt.model_name "$LLM_MODEL"
  echo ""
  echo "Simulstream output written to: $HYPOTHESIS_JSON"
fi

if [[ -n "$SIMULSTREAM_CONFIG" ]]; then
  if [[ -z "$SEGMENTS_MANIFEST_ABS" ]]; then
     echo "Error: --segments-manifest is required for omnisteval evaluation."
     exit 1
  fi

  SPEECH_SEGMENTATION="$OUTPUT_DIR_ABS/audio_definitions.yaml"
  echo ""
  echo "========== 3. Run omnisteval longform =========="
  echo "Using generated speech segmentation: $SPEECH_SEGMENTATION"
  
  OMNI_OUTPUT="$OUTPUT_DIR_ABS/omnisteval"
  python -m omnisteval.cli longform \
    --speech_segmentation "$SPEECH_SEGMENTATION" \
    --ref_sentences_file "$OUTPUT_DIR_ABS/references.txt" \
    --hypothesis_file "$HYPOTHESIS_JSON" \
    --hypothesis_format=simulstream \
    --simulstream_config_file "$SIMULSTREAM_CONFIG" \
    --lang "$TGT_LANG" \
    --source_sentences_file "$OUTPUT_DIR_ABS/transcripts.txt" \
    --output_folder "$OMNI_OUTPUT" \
    $COMET
  echo "Omnisteval results in: $OMNI_OUTPUT"
else
  echo "Omnisteval skipped (set --simulstream-config to run)."
fi

echo ""
echo "Done. Output directory: $OUTPUT_DIR_ABS"
