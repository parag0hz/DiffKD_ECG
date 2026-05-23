#!/bin/bash
# Fair MVKT comparison: Lead I 100Hz student, same setting as MVKT-ECG.
#
# Trains student variants with different KD loss combinations on Lead I 100Hz.
# This provides an apples-to-apples comparison with MVKT-ECG (AUC=0.843, F1=0.626).
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_mvkt_fair_100hz_leadI.sh
#
# Environment variables:
#   DATA_DIR       path to PTB-XL root
#   TEACHER_CKPT   teacher checkpoint
#   TA_CKPT        TA-I-500Hz checkpoint
#   SEED           random seed (default: 0)
#   BATCH_SIZE     batch size (default: 256)
#   OUT_DIR        output directory (default: dafd_mvkt/outputs)

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
TA_CKPT=${TA_CKPT:-"dafd_mvkt/outputs/ta_I_500hz_seed0_best.pt"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}

echo "========================================================"
echo "MVKT Fair Comparison  Lead=I  100Hz  seed=${SEED}"
echo "DATA_DIR=${DATA_DIR}"
echo "TEACHER=${TEACHER_CKPT}"
echo "TA=${TA_CKPT}"
echo "========================================================"

run_student() {
    local LOSSES=$1
    local LOSSES_FNAME
    LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
    local MODEL_NAME="student_I_100hz_${LOSSES_FNAME}_seed${SEED}"
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
            --config dafd_mvkt/configs/student_100hz_leadI.yaml \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --ta_ckpt "${TA_CKPT}" \
            --ta_config dafd_mvkt/configs/ta_i_500hz.yaml \
            --lead "I" \
            --losses "${LOSSES}" \
            --seed "${SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --run_name "${MODEL_NAME}" \
            --output_dir "${OUT_DIR}"
    fi

    echo "  Evaluating …"
    if [ -f "${METRICS}" ]; then
        echo "  Metrics exist — skipping evaluation."
    else
        python dafd_mvkt/evaluate.py \
            --config dafd_mvkt/configs/student_100hz_leadI.yaml \
            --data_dir "${DATA_DIR}" \
            --ckpt "${CKPT}" \
            --split test \
            --model_name "${MODEL_NAME}" \
            --output_dir "${OUT_DIR}" \
            --tune_thresholds
    fi
}

# Step 1: Train TA-I-500Hz if not present
TA_METRICS="${OUT_DIR}/metrics_test_ta_I_500hz_seed${SEED}.json"
echo ""
echo "  Checking TA-I-500Hz …"
if [ ! -f "${TA_CKPT}" ]; then
    echo "  Training TA-I-500Hz …"
    python dafd_mvkt/train_ta.py \
        --config dafd_mvkt/configs/ta_i_500hz.yaml \
        --data_dir "${DATA_DIR}" \
        --teacher_ckpt "${TEACHER_CKPT}" \
        --lead "I" \
        --batch_size "${BATCH_SIZE}" \
        --run_name "ta_I_500hz_seed${SEED}" \
        --output_dir "${OUT_DIR}"
fi
if [ -f "${TA_METRICS}" ]; then
    echo "  TA metrics exist — skipping evaluation."
else
    python dafd_mvkt/evaluate.py \
        --config dafd_mvkt/configs/ta_i_500hz.yaml \
        --data_dir "${DATA_DIR}" \
        --ckpt "${TA_CKPT}" \
        --split test \
        --model_name "ta_I_500hz_seed${SEED}" \
        --output_dir "${OUT_DIR}" \
        --tune_thresholds
fi

# Step 2: Student variants (BCE → +ta_mkd → full HST-KD)
run_student "bce"
run_student "bce,ta_mkd"
run_student "bce,ta_mkd,ta_crf,feature"

echo ""
echo "MVKT fair comparison (Lead I 100Hz) complete."
echo "Results in ${OUT_DIR}"
