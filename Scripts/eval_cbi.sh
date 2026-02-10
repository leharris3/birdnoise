#!/usr/bin/env bash
# eval_cbi.sh — Fast evaluation of fusion checkpoints on CBI (BirdCLEF 2021).
# Assumes a GPU is already available (e.g. inside an interactive salloc session).
# Usage:  bash Scripts/eval_cbi.sh

set -euo pipefail

# ── Paths (edit if your layout differs) ──────────────────────────
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DATA_DIR="${ROOT}/Data/beans/cbi"
PRIORS_CACHE="${ROOT}/Data/beans/cbi/priors_cache_temp_1766584745.h5"

CKPT_STAGE_A="${ROOT}/exps/checkpoints_fusion_stage_a/best_model.pth"
CKPT_STAGE_B="${ROOT}/exps/checkpoints_fusion_stage_b/best_model.pth"

# ── Sanity checks ────────────────────────────────────────────────
for f in "$DATA_DIR/train.csv" "$PRIORS_CACHE" "$CKPT_STAGE_A" "$CKPT_STAGE_B"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing file: $f" >&2
        exit 1
    fi
done

echo "=== Evaluating on CBI ==="
echo "  Data dir:       $DATA_DIR"
echo "  Priors cache:   $PRIORS_CACHE"
echo "  Checkpoints:    $CKPT_STAGE_A"
echo "                  $CKPT_STAGE_B"
echo ""

# ── Run evaluation ───────────────────────────────────────────────
cd "${ROOT}/Scripts"

uv run python evaluate_models_fast.py \
    --data_dir      "$DATA_DIR" \
    --priors_cache  "$PRIORS_CACHE" \
    --checkpoints   "$CKPT_STAGE_A" "$CKPT_STAGE_B" \
    --batch_size    64 \
    --num_workers   32 \
    --device        cuda \
    --pooling       mean \
    --seed          42

echo ""
echo "=== Done. Results saved to eval_results.pth ==="