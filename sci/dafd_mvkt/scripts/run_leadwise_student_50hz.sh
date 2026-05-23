#!/bin/bash
# Lead-wise 50Hz student training.
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl LEADS="I II V1 V2 V5" bash dafd_mvkt/scripts/run_leadwise_student_50hz.sh

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
LEADS=${LEADS:-"I II V1 V2 V5"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
HZ=50
CONFIG="dafd_mvkt/configs/student_50hz_gated.yaml"

echo "========================================================"
echo "Lead-wise 50Hz student  leads=${LEADS}  seed=${SEED}"
echo "========================================================"

for LEAD in ${LEADS}; do
    TA_CKPT="${OUT_DIR}/ta_${LEAD}_500hz_seed${SEED}_best.pt"

    if [ ! -f "${TA_CKPT}" ]; then
        if [ "${LEAD}" = "II" ]; then
            TA_CKPT="outputs/ta_ii_500hz_best.pt"
        else
            echo "  [skip] TA checkpoint not found for ${LEAD}: ${TA_CKPT}"
            continue
        fi
    fi

    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  Lead: ${LEAD}  50Hz                          ║"
    echo "╚══════════════════════════════════════════╝"

    for LOSSES in "bce" "bce,ta_mkd" "bce,ta_mkd,ta_crf,feature" "bce,gated_kd,ta_crf,feature"; do
        LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
        MODEL_NAME="student_${LEAD}_${HZ}hz_${LOSSES_FNAME}_seed${SEED}"
        CKPT="${OUT_DIR}/${MODEL_NAME}_best.pt"
        METRICS="${OUT_DIR}/metrics_test_${MODEL_NAME}.json"

        echo ""
        echo "  Train: ${MODEL_NAME}"

        if [ -f "${CKPT}" ]; then
            echo "  Checkpoint exists — skipping."
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

        if [ -f "${METRICS}" ]; then
            echo "  Metrics exist — skipping eval."
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
    done
done

echo ""
echo "All lead-wise 50Hz student runs complete."
