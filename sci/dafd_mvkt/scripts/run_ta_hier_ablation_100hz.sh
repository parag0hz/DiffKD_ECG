#!/bin/bash
# Sequential TA-hier ablation for 100Hz student.
# Runs 6 loss configurations one at a time to maximize GPU VRAM per run.
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_ta_hier_ablation_100hz.sh
#
# Environment variables:
#   DATA_DIR       path to PTB-XL root
#   TEACHER_CKPT   teacher checkpoint (default: dafd_mvkt/outputs/teacher_best.pt)
#   TA_CKPT        TA checkpoint      (default: outputs/ta_ii_500hz_best.pt)
#   LEAD           ECG lead           (default: II)
#   SEED           random seed        (default: 0)
#   BATCH_SIZE     batch size         (default: 128)
#   OUT_DIR        output directory   (default: dafd_mvkt/outputs)

set -e
cd "$(dirname "$0")/../.."   # ensure we run from ~/sci

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
TA_CKPT=${TA_CKPT:-"outputs/ta_ii_500hz_best.pt"}
LEAD=${LEAD:-"II"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
CONFIG="dafd_mvkt/configs/student_100hz_ta.yaml"
HZ=100

echo "========================================================"
echo "TA-hier 100Hz ablation  lead=${LEAD}  seed=${SEED}"
echo "DATA_DIR=${DATA_DIR}"
echo "TEACHER=${TEACHER_CKPT}"
echo "TA=${TA_CKPT}"
echo "BATCH_SIZE=${BATCH_SIZE}"
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
run_one "bce,ta_mkd"
run_one "bce,ta_mkd,ta_crf"
run_one "bce,ta_mkd,feature"
run_one "bce,ta_mkd,ta_crf,feature"
run_one "bce,teacher_mkd,ta_mkd,ta_crf,feature"

echo ""
echo "All 100Hz ablation runs complete."
