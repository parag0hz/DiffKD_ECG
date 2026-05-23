#!/bin/bash
# Temporal Segment Feature KD ablation for 100Hz student.
# Compares: global FeatureKD vs SegmentFeatureKD (uniform) vs SegmentFeatureKD (attention).
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_segment_feature_100hz.sh

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
TA_CKPT=${TA_CKPT:-"outputs/ta_ii_500hz_best.pt"}
LEAD=${LEAD:-"II"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
CONFIG="dafd_mvkt/configs/student_100hz_segment.yaml"
HZ=100

echo "========================================================"
echo "Segment Feature KD 100Hz  lead=${LEAD}  seed=${SEED}"
echo "DATA_DIR=${DATA_DIR}  BATCH_SIZE=${BATCH_SIZE}"
echo "========================================================"

run_one() {
    local LOSSES=$1
    local MODEL_NAME=$2
    local EXTRA_ARGS=$3
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
            --config "${CONFIG}" \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --ta_ckpt "${TA_CKPT}" \
            --lead "${LEAD}" \
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
            --config "${CONFIG}" \
            --data_dir "${DATA_DIR}" \
            --ckpt "${CKPT}" \
            --split test \
            --model_name "${MODEL_NAME}" \
            --output_dir "${OUT_DIR}" \
            --tune_thresholds
    fi
}

# Reference: global FeatureKD (reuses existing ckpt if present)
GLOBAL_NAME="student_${LEAD}_${HZ}hz_bce_ta_mkd_ta_crf_feature_seed${SEED}"
run_one "bce,ta_mkd,ta_crf,feature" "${GLOBAL_NAME}"

# Segment FeatureKD (uniform weights)
SEG_NAME="student_${LEAD}_${HZ}hz_bce_ta_mkd_ta_crf_segment_feature_seed${SEED}"
run_one "bce,ta_mkd,ta_crf,segment_feature" "${SEG_NAME}"

# Segment FeatureKD (attention-weighted)
SEG_ATTN_NAME="student_${LEAD}_${HZ}hz_bce_ta_mkd_ta_crf_segment_feature_attn_seed${SEED}"
run_one "bce,ta_mkd,ta_crf,segment_feature" "${SEG_ATTN_NAME}" "--attention_weighted_segment"

echo ""
echo "All Segment Feature KD 100Hz runs complete."
