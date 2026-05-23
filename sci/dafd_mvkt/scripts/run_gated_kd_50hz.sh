#!/bin/bash
# Confidence-Gated Hierarchical KD ablation for 50Hz student.
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_gated_kd_50hz.sh

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
TA_CKPT=${TA_CKPT:-"outputs/ta_ii_500hz_best.pt"}
LEAD=${LEAD:-"II"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
GATE_MODE=${GATE_MODE:-"confidence_agreement"}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
CONFIG="dafd_mvkt/configs/student_50hz_gated.yaml"
HZ=50

echo "========================================================"
echo "Gated KD 50Hz ablation  lead=${LEAD}  seed=${SEED}"
echo "DATA_DIR=${DATA_DIR}  BATCH_SIZE=${BATCH_SIZE}"
echo "GATE_MODE=${GATE_MODE}"
echo "========================================================"

run_one() {
    local LOSSES=$1
    local LOSSES_FNAME
    LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
    local MODEL_NAME="student_${LEAD}_${HZ}hz_${LOSSES_FNAME}_seed${SEED}"
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
            --gate_mode "${GATE_MODE}" \
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

run_one "bce"
run_one "bce,ta_mkd,ta_crf,feature"
run_one "bce,gated_kd"
run_one "bce,gated_kd,ta_crf"
run_one "bce,gated_kd,feature"
run_one "bce,gated_kd,ta_crf,feature"

echo ""
echo "All Gated KD 50Hz runs complete."
