#!/bin/bash
# Progressive Temporal TA distillation for 50Hz student.
#
# Steps:
#   1. Train 1-lead 100Hz TA from frozen 500Hz TA
#   2. Evaluate 100Hz TA on test set
#   3. Train 50Hz student from 100Hz TA (bce,ta_mkd,ta_crf,feature)
#   4. Compare: bce baseline and TA500-hier reference
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_progressive_50hz.sh
#
# Environment variables:
#   DATA_DIR       path to PTB-XL root
#   TEACHER_CKPT   teacher checkpoint (default: dafd_mvkt/outputs/teacher_best.pt)
#   TA500_CKPT     500Hz TA checkpoint (default: outputs/ta_ii_500hz_best.pt)
#   LEAD           ECG lead           (default: II)
#   SEED           random seed        (default: 0)
#   BATCH_SIZE     batch size         (default: 256)
#   OUT_DIR        output directory   (default: dafd_mvkt/outputs)

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
TA500_CKPT=${TA500_CKPT:-"outputs/ta_ii_500hz_best.pt"}
LEAD=${LEAD:-"II"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}

TA100_NAME="ta_${LEAD}_100hz_from_ta500_seed${SEED}"
TA100_CKPT="${OUT_DIR}/${TA100_NAME}_best.pt"

echo "========================================================"
echo "Progressive TA 50Hz  lead=${LEAD}  seed=${SEED}"
echo "DATA_DIR=${DATA_DIR}  BATCH_SIZE=${BATCH_SIZE}"
echo "TA500=${TA500_CKPT}"
echo "TA100=${TA100_CKPT}"
echo "========================================================"

# ── Step 1: Train 100Hz TA ────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  Step 1: Train 100Hz progressive TA"
echo "════════════════════════════════════════"

if [ -f "${TA100_CKPT}" ]; then
    echo "  Checkpoint exists — skipping TA100 training."
else
    python dafd_mvkt/train_temporal_ta.py \
        --config dafd_mvkt/configs/ta_ii_100hz_progressive.yaml \
        --data_dir "${DATA_DIR}" \
        --high_ta_ckpt "${TA500_CKPT}" \
        --lead "${LEAD}" \
        --seed "${SEED}" \
        --batch_size "${BATCH_SIZE}" \
        --output_dir "${OUT_DIR}"
fi

# ── Step 2: Evaluate 100Hz TA ─────────────────────────────────────────────────
TA100_METRICS="${OUT_DIR}/metrics_test_${TA100_NAME}.json"
echo ""
echo "  Evaluating 100Hz TA …"
if [ -f "${TA100_METRICS}" ]; then
    echo "  Metrics exist — skipping."
else
    python dafd_mvkt/evaluate.py \
        --config dafd_mvkt/configs/ta_ii_100hz_progressive.yaml \
        --data_dir "${DATA_DIR}" \
        --ckpt "${TA100_CKPT}" \
        --split test \
        --model_name "${TA100_NAME}" \
        --output_dir "${OUT_DIR}" \
        --tune_thresholds
fi

# ── Step 3: Train 50Hz student from 100Hz TA ─────────────────────────────────
run_student() {
    local LOSSES=$1
    local EXTRA_ARGS=$2
    local LOSSES_FNAME
    LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
    # Append _prog suffix to distinguish from TA500-based runs
    local MODEL_NAME="student_${LEAD}_50hz_${LOSSES_FNAME}_prog_seed${SEED}"
    local CKPT="${OUT_DIR}/${MODEL_NAME}_best.pt"
    local METRICS="${OUT_DIR}/metrics_test_${MODEL_NAME}.json"

    echo ""
    echo "════════════════════════════════════════"
    echo "  Train: ${MODEL_NAME}"
    echo "════════════════════════════════════════"

    if [ -f "${CKPT}" ]; then
        echo "  Checkpoint exists — skipping training."
    else
        python dafd_mvkt/train_student_hier.py \
            --config dafd_mvkt/configs/student_50hz_progressive.yaml \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --ta_ckpt "${TA100_CKPT}" \
            --ta_config dafd_mvkt/configs/ta_ii_100hz_progressive.yaml \
            --lead "${LEAD}" \
            --losses "${LOSSES}" \
            --ta_hz 100 \
            --seed "${SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --run_name "${MODEL_NAME}" \
            --output_dir "${OUT_DIR}" \
            ${EXTRA_ARGS}
    fi

    echo "  Evaluating …"
    if [ -f "${METRICS}" ]; then
        echo "  Metrics exist — skipping evaluation."
    else
        python dafd_mvkt/evaluate.py \
            --config dafd_mvkt/configs/student_50hz_progressive.yaml \
            --data_dir "${DATA_DIR}" \
            --ckpt "${CKPT}" \
            --split test \
            --model_name "${MODEL_NAME}" \
            --output_dir "${OUT_DIR}" \
            --tune_thresholds
    fi
}

echo ""
echo "══ Step 3: 50Hz student from 100Hz TA ══"
run_student "bce"
run_student "bce,ta_mkd"
run_student "bce,ta_mkd,ta_crf,feature"

echo ""
echo "All progressive 50Hz runs complete."
echo "Compare with TA500-based results in ${OUT_DIR}"
