#!/bin/bash
# CLECG + HST-KD experiments: Lead II 100Hz student.
#
# Experiment ladder:
#   1. BCE baseline (no KD, no pretrain)
#   2. HST-KD (bce,ta_mkd,ta_crf,feature) — standard
#   3. CLECG pretrain → BCE fine-tune
#   4. CLECG pretrain → HST-KD fine-tune
#   5. CLECG pretrain → TA500 + HST-KD fine-tune (full pipeline)
#
# CLECG encoder init: 100Hz encoder (same Hz as student).
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_clecg_hst_100hz_leadII.sh
#
# Environment variables:
#   DATA_DIR            path to PTB-XL root
#   TEACHER_CKPT        teacher checkpoint
#   TA_CKPT             TA-II-500Hz checkpoint
#   CLECG_ENCODER_CKPT  CLECG encoder checkpoint (100Hz Lead II)
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
echo "CLECG + HST-KD  Lead=II  100Hz  seed=${SEED}"
echo "DATA_DIR=${DATA_DIR}"
echo "TEACHER=${TEACHER_CKPT}"
echo "TA=${TA_CKPT}"
echo "CLECG_ENCODER=${CLECG_ENCODER_CKPT}"
echo "========================================================"

run_student() {
    local LOSSES=$1
    local SUFFIX=$2          # extra suffix for run name (e.g. "clecg")
    local INIT_ENC=$3        # optional encoder init ckpt

    local LOSSES_FNAME
    LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
    local MODEL_NAME="student_II_100hz_${LOSSES_FNAME}${SUFFIX}_seed${SEED}"
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
            --config dafd_mvkt/configs/student_100hz_ta.yaml \
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
            --config dafd_mvkt/configs/student_100hz_ta.yaml \
            --data_dir "${DATA_DIR}" \
            --ckpt "${CKPT}" \
            --split test \
            --model_name "${MODEL_NAME}" \
            --output_dir "${OUT_DIR}" \
            --tune_thresholds
    fi
}

# Ensure CLECG encoder exists
if [ ! -f "${CLECG_ENCODER_CKPT}" ]; then
    echo "WARNING: CLECG encoder not found at ${CLECG_ENCODER_CKPT}"
    echo "Run run_clecg_pretrain.sh first, or set CLECG_ENCODER_CKPT."
    echo "Skipping CLECG init experiments."
    SKIP_CLECG=1
else
    SKIP_CLECG=0
fi

# 1. BCE baseline
run_student "bce" "" ""

# 2. HST-KD (standard, no pretrain)
run_student "bce,ta_mkd,ta_crf,feature" "" ""

if [ "${SKIP_CLECG}" = "0" ]; then
    # 3. CLECG → BCE
    run_student "bce" "_clecg" "${CLECG_ENCODER_CKPT}"

    # 4. CLECG → HST-KD
    run_student "bce,ta_mkd,ta_crf,feature" "_clecg" "${CLECG_ENCODER_CKPT}"
fi

echo ""
echo "CLECG+HST-KD Lead II 100Hz complete."
echo "Results in ${OUT_DIR}"
