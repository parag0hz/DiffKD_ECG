#!/bin/bash
# 50Hz trade-off experiments: Lead II student.
#
# Compares: BCE / HST-KD / CLECG+HST-KD using direct TA500 distillation.
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_50hz_tradeoff_leadII.sh
#
# Environment variables:
#   DATA_DIR            path to PTB-XL root
#   TEACHER_CKPT        teacher checkpoint
#   TA_CKPT             TA-II-500Hz checkpoint
#   CLECG_ENCODER_CKPT  CLECG encoder checkpoint (50Hz Lead II, or 100Hz as fallback)
#   SEED                random seed (default: 0)
#   BATCH_SIZE          batch size (default: 256)
#   OUT_DIR             output directory (default: dafd_mvkt/outputs)

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
TA_CKPT=${TA_CKPT:-"dafd_mvkt/outputs/ta_II_500hz_seed0_best.pt"}
CLECG_ENCODER_CKPT=${CLECG_ENCODER_CKPT:-"dafd_mvkt/outputs/clecg_II_100hz_seed0_encoder.pt"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}

echo "========================================================"
echo "50Hz Trade-off  Lead=II  seed=${SEED}"
echo "TA=${TA_CKPT}"
echo "========================================================"

run_student() {
    local LOSSES=$1
    local SUFFIX=$2
    local INIT_ENC=$3

    local LOSSES_FNAME
    LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
    local MODEL_NAME="student_II_50hz_${LOSSES_FNAME}${SUFFIX}_seed${SEED}"
    local CKPT="${OUT_DIR}/${MODEL_NAME}_best.pt"
    local METRICS="${OUT_DIR}/metrics_test_${MODEL_NAME}.json"

    echo ""
    echo "════════════════════════════════════════"
    echo "  Train: ${MODEL_NAME}"
    echo "════════════════════════════════════════"

    local EXTRA_ARGS=""
    if [ -n "${INIT_ENC}" ] && [ -f "${INIT_ENC}" ]; then
        EXTRA_ARGS="--init_encoder_ckpt ${INIT_ENC}"
    fi

    if [ -f "${CKPT}" ]; then
        echo "  Checkpoint exists — skipping training."
    else
        python dafd_mvkt/train_student_hier.py \
            --config dafd_mvkt/configs/student_50hz_ta.yaml \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --ta_ckpt "${TA_CKPT}" \
            --lead "II" \
            --losses "${LOSSES}" \
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
            --config dafd_mvkt/configs/student_50hz_ta.yaml \
            --data_dir "${DATA_DIR}" \
            --ckpt "${CKPT}" \
            --split test \
            --model_name "${MODEL_NAME}" \
            --output_dir "${OUT_DIR}" \
            --tune_thresholds
    fi
}

if [ ! -f "${CLECG_ENCODER_CKPT}" ]; then
    echo "WARNING: CLECG encoder not found: ${CLECG_ENCODER_CKPT}"
    SKIP_CLECG=1
else
    SKIP_CLECG=0
fi

run_student "bce" "" ""
run_student "bce,ta_mkd,ta_crf,feature" "" ""

if [ "${SKIP_CLECG}" = "0" ]; then
    run_student "bce,ta_mkd,ta_crf,feature" "_clecg" "${CLECG_ENCODER_CKPT}"
fi

echo ""
echo "50Hz trade-off Lead II complete."
