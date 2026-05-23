#!/bin/bash
# Multi-seed TA-hier experiments (100Hz + 50Hz, sequential).
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl SEEDS="0 1 2 3 4" bash dafd_mvkt/scripts/run_multiseed_ta_hier.sh
#
# Environment variables:
#   DATA_DIR       path to PTB-XL root
#   TEACHER_CKPT   teacher checkpoint (default: dafd_mvkt/outputs/teacher_best.pt)
#   TA_CKPT        TA checkpoint      (default: outputs/ta_ii_500hz_best.pt)
#   LEAD           ECG lead           (default: II)
#   SEEDS          space-separated seeds (default: "0 1 2 3 4")
#   BATCH_SIZE     batch size         (default: 128)
#   RUN_FULL       set to 1 to include bce,teacher_mkd,ta_mkd,ta_crf,feature
#   OUT_DIR        output directory   (default: dafd_mvkt/outputs)

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
TA_CKPT=${TA_CKPT:-"outputs/ta_ii_500hz_best.pt"}
LEAD=${LEAD:-"II"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
BATCH_SIZE=${BATCH_SIZE:-256}
RUN_FULL=${RUN_FULL:-0}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}

ABLATIONS="bce bce,ta_mkd bce,ta_mkd,ta_crf bce,ta_mkd,feature bce,ta_mkd,ta_crf,feature"
if [ "${RUN_FULL}" = "1" ]; then
    ABLATIONS="${ABLATIONS} bce,teacher_mkd,ta_mkd,ta_crf,feature"
fi

echo "========================================================"
echo "Multi-seed TA-hier  lead=${LEAD}  seeds=${SEEDS}"
echo "DATA_DIR=${DATA_DIR}  BATCH_SIZE=${BATCH_SIZE}"
echo "RUN_FULL=${RUN_FULL}"
echo "Ablations: ${ABLATIONS}"
echo "========================================================"

train_eval() {
    local CONFIG=$1
    local LOSSES=$2
    local SEED=$3
    local HZ=$4

    local LOSSES_FNAME
    LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
    local MODEL_NAME="student_${LEAD}_${HZ}hz_${LOSSES_FNAME}_seed${SEED}"
    local CKPT="${OUT_DIR}/${MODEL_NAME}_best.pt"
    local METRICS="${OUT_DIR}/metrics_test_${MODEL_NAME}.json"

    if [ -f "${CKPT}" ]; then
        echo "  [skip train] ${MODEL_NAME}"
    else
        echo "  [train] ${MODEL_NAME}"
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

    if [ -f "${METRICS}" ]; then
        echo "  [skip eval] ${MODEL_NAME}"
    else
        echo "  [eval]  ${MODEL_NAME}"
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

for SEED in ${SEEDS}; do
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  SEED ${SEED}                                   ║"
    echo "╚══════════════════════════════════════════╝"

    for LOSSES in ${ABLATIONS}; do
        train_eval "dafd_mvkt/configs/student_100hz_ta.yaml" "${LOSSES}" "${SEED}" "100"
    done

    for LOSSES in ${ABLATIONS}; do
        train_eval "dafd_mvkt/configs/student_50hz_ta.yaml" "${LOSSES}" "${SEED}" "50"
    done
done

echo ""
echo "All multi-seed runs complete."
echo "Run aggregate: python dafd_mvkt/experiments/aggregate_results.py --results_dir ${OUT_DIR}"
