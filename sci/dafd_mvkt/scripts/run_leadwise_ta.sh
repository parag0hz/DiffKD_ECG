#!/bin/bash
# Train 1-lead 500Hz TA for multiple leads.
# Reuses existing checkpoints.
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl LEADS="I II V1 V2 V5" bash dafd_mvkt/scripts/run_leadwise_ta.sh

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
LEADS=${LEADS:-"I II V1 V2 V5"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}

echo "========================================================"
echo "Lead-wise TA training  leads=${LEADS}  seed=${SEED}"
echo "DATA_DIR=${DATA_DIR}  BATCH_SIZE=${BATCH_SIZE}"
echo "========================================================"

for LEAD in ${LEADS}; do
    TA_NAME="ta_${LEAD}_500hz_seed${SEED}"
    TA_CKPT="${OUT_DIR}/${TA_NAME}_best.pt"
    TA_METRICS="${OUT_DIR}/metrics_test_${TA_NAME}.json"

    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  Lead: ${LEAD}                                  ║"
    echo "╚══════════════════════════════════════════╝"

    if [ -f "${TA_CKPT}" ]; then
        echo "  TA checkpoint exists — skipping training."
    else
        python dafd_mvkt/train_ta.py \
            --config dafd_mvkt/configs/ta_ii_500hz.yaml \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --lead "${LEAD}" \
            --seed "${SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --run_name "${TA_NAME}" \
            --output_dir "${OUT_DIR}"
    fi

    echo "  Evaluating TA for lead ${LEAD} …"
    if [ -f "${TA_METRICS}" ]; then
        echo "  Metrics exist — skipping."
    else
        python dafd_mvkt/evaluate.py \
            --config dafd_mvkt/configs/ta_ii_500hz.yaml \
            --data_dir "${DATA_DIR}" \
            --ckpt "${TA_CKPT}" \
            --split test \
            --model_name "${TA_NAME}" \
            --output_dir "${OUT_DIR}" \
            --lead "${LEAD}" \
            --tune_thresholds
    fi
done

echo ""
echo "All lead-wise TA runs complete."
