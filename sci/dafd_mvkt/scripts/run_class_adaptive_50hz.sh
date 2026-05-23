#!/bin/bash
# Class-Adaptive KD for 50Hz student.
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_class_adaptive_50hz.sh

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
TA_CKPT=${TA_CKPT:-"outputs/ta_ii_500hz_best.pt"}
LEAD=${LEAD:-"II"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
WEIGHT_MODE=${WEIGHT_MODE:-"inverse_baseline_auc"}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
HZ=50
CONFIG="dafd_mvkt/configs/student_50hz_gated.yaml"
WEIGHTS_DIR="${OUT_DIR}/class_weights"

echo "========================================================"
echo "Class-Adaptive KD 50Hz  lead=${LEAD}  seed=${SEED}"
echo "WEIGHT_MODE=${WEIGHT_MODE}"
echo "========================================================"

BCE_MODEL="student_${LEAD}_${HZ}hz_bce_seed${SEED}"
BCE_CKPT="${OUT_DIR}/${BCE_MODEL}_best.pt"
BCE_VAL_METRICS="${OUT_DIR}/metrics_val_${BCE_MODEL}.json"

echo ""
echo "  Step 1: Baseline val metrics"
if [ -f "${BCE_VAL_METRICS}" ]; then
    echo "  Already exists: ${BCE_VAL_METRICS}"
else
    if [ ! -f "${BCE_CKPT}" ]; then
        echo "  ERROR: Baseline checkpoint not found: ${BCE_CKPT}"
        echo "  Run run_ta_hier_ablation_50hz.sh first."
        exit 1
    fi
    python dafd_mvkt/evaluate.py \
        --config dafd_mvkt/configs/student_50hz_ta.yaml \
        --data_dir "${DATA_DIR}" \
        --ckpt "${BCE_CKPT}" \
        --split val \
        --model_name "${BCE_MODEL}" \
        --output_dir "${OUT_DIR}"
fi

WEIGHTS_JSON="${WEIGHTS_DIR}/${LEAD}_${HZ}hz_${WEIGHT_MODE}.json"
echo ""
echo "  Step 2: Computing class weights (mode=${WEIGHT_MODE})"
if [ -f "${WEIGHTS_JSON}" ]; then
    echo "  Already exists: ${WEIGHTS_JSON}"
else
    python dafd_mvkt/experiments/compute_class_weights.py \
        --mode "${WEIGHT_MODE}" \
        --baseline_metrics "${BCE_VAL_METRICS}" \
        --lead "${LEAD}" --hz "${HZ}" \
        --out "${WEIGHTS_JSON}"
fi

run_adaptive() {
    local LOSSES=$1
    local LOSSES_FNAME
    LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
    local MODEL_NAME="student_${LEAD}_${HZ}hz_${LOSSES_FNAME}_cadap_seed${SEED}"
    local CKPT="${OUT_DIR}/${MODEL_NAME}_best.pt"
    local METRICS="${OUT_DIR}/metrics_test_${MODEL_NAME}.json"

    echo ""
    echo "════════════════════════════════════════"
    echo "  Train (class-adaptive): ${MODEL_NAME}"
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
            --class_weights_json "${WEIGHTS_JSON}" \
            --run_name "${MODEL_NAME}" \
            --output_dir "${OUT_DIR}"
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

run_adaptive "bce,ta_mkd,ta_crf,feature"
run_adaptive "bce,gated_kd,ta_crf,feature"

echo ""
echo "All class-adaptive 50Hz runs complete."
