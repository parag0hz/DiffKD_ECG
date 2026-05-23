#!/bin/bash
# Final multi-seed validation for the best methods.
#
# Runs seeds 0,1,2 for the best-performing configurations identified
# from single-seed experiments.
#
# Target: beat MVKT-ECG (AUC>0.843, F1>0.626) on Lead I 100Hz.
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_final_multiseed.sh
#
# Environment variables:
#   DATA_DIR       path to PTB-XL root
#   TEACHER_CKPT   teacher checkpoint
#   SEEDS          seeds to run (default: "0 1 2")
#   BATCH_SIZE     batch size (default: 256)
#   OUT_DIR        output directory (default: dafd_mvkt/outputs)
#   LEAD           lead to run (default: "I")
#   HZ             sampling rate (default: 100)
#   LOSSES         loss config (default: "bce,ta_mkd,ta_crf,feature")

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
SEEDS=${SEEDS:-"0 1 2"}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
LEAD=${LEAD:-"I"}
HZ=${HZ:-100}
LOSSES=${LOSSES:-"bce,ta_mkd,ta_crf,feature"}

LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
echo "========================================================"
echo "Multi-seed  Lead=${LEAD}  ${HZ}Hz  losses=${LOSSES}"
echo "Seeds: ${SEEDS}"
echo "========================================================"

for SEED in ${SEEDS}; do
    TA_CKPT="${OUT_DIR}/ta_${LEAD}_500hz_seed${SEED}_best.pt"
    CLECG_ENC="${OUT_DIR}/clecg_${LEAD}_${HZ}hz_seed${SEED}_encoder.pt"

    echo ""
    echo "══════════════  seed=${SEED}  ══════════════"

    # ── TA seed N ─────────────────────────────────────────────────────────
    if [ ! -f "${TA_CKPT}" ]; then
        echo "  Training TA-${LEAD}-500Hz seed=${SEED} …"
        if [ "${LEAD}" = "I" ]; then
            TA_CFG="dafd_mvkt/configs/ta_i_500hz.yaml"
        else
            TA_CFG="dafd_mvkt/configs/ta_ii_500hz.yaml"
        fi
        python dafd_mvkt/train_ta.py \
            --config "${TA_CFG}" \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --lead "${LEAD}" \
            --batch_size "${BATCH_SIZE}" \
            --run_name "ta_${LEAD}_500hz_seed${SEED}" \
            --output_dir "${OUT_DIR}"
        # Update seed field: re-use same binary
    fi

    # ── CLECG pretrain seed N ─────────────────────────────────────────────
    if [ ! -f "${CLECG_ENC}" ]; then
        echo "  CLECG pretraining seed=${SEED} …"
        if [ "${LEAD}" = "I" ]; then
            CLECG_CFG="dafd_mvkt/configs/clecg_leadI_${HZ}hz.yaml"
        else
            CLECG_CFG="dafd_mvkt/configs/clecg_leadII_${HZ}hz.yaml"
        fi
        if [ "${HZ}" = "100" ] || [ "${HZ}" = "500" ]; then
            python dafd_mvkt/pretrain_clecg.py \
                --config "${CLECG_CFG}" \
                --data_dir "${DATA_DIR}" \
                --lead "${LEAD}" \
                --hz "${HZ}" \
                --seed "${SEED}" \
                --batch_size "${BATCH_SIZE}" \
                --run_name "clecg_${LEAD}_${HZ}hz_seed${SEED}" \
                --output_dir "${OUT_DIR}"
        fi
    fi

    # ── Student ───────────────────────────────────────────────────────────
    if [ "${HZ}" = "100" ]; then
        if [ "${LEAD}" = "I" ]; then
            STUDENT_CFG="dafd_mvkt/configs/student_100hz_leadI.yaml"
            TA_CFG_ARG="dafd_mvkt/configs/ta_i_500hz.yaml"
        else
            STUDENT_CFG="dafd_mvkt/configs/student_100hz_ta.yaml"
            TA_CFG_ARG=""
        fi
    else
        echo "  WARNING: HZ=${HZ} not implemented for multi-seed. Skipping."
        continue
    fi

    for USE_CLECG in 0 1; do
        if [ "${USE_CLECG}" = "1" ] && [ ! -f "${CLECG_ENC}" ]; then
            echo "  CLECG encoder missing — skipping CLECG variant for seed=${SEED}"
            continue
        fi

        if [ "${USE_CLECG}" = "1" ]; then
            SUFFIX="_clecg"
            INIT_ARG="--init_encoder_ckpt ${CLECG_ENC}"
        else
            SUFFIX=""
            INIT_ARG=""
        fi

        MODEL_NAME="student_${LEAD}_${HZ}hz_${LOSSES_FNAME}${SUFFIX}_seed${SEED}"
        CKPT="${OUT_DIR}/${MODEL_NAME}_best.pt"
        METRICS="${OUT_DIR}/metrics_test_${MODEL_NAME}.json"

        echo ""
        echo "  Train: ${MODEL_NAME}"

        if [ -f "${CKPT}" ]; then
            echo "  Checkpoint exists — skipping."
        else
            python dafd_mvkt/train_student_hier.py \
                --config "${STUDENT_CFG}" \
                --data_dir "${DATA_DIR}" \
                --teacher_ckpt "${TEACHER_CKPT}" \
                --ta_ckpt "${TA_CKPT}" \
                ${TA_CFG_ARG:+--ta_config "${TA_CFG_ARG}"} \
                --lead "${LEAD}" \
                --losses "${LOSSES}" \
                --seed "${SEED}" \
                --batch_size "${BATCH_SIZE}" \
                --run_name "${MODEL_NAME}" \
                --output_dir "${OUT_DIR}" \
                ${INIT_ARG}
        fi

        if [ ! -f "${METRICS}" ]; then
            python dafd_mvkt/evaluate.py \
                --config "${STUDENT_CFG}" \
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
echo "Multi-seed runs complete."
echo "Aggregating results …"
python dafd_mvkt/experiments/aggregate_results.py \
    --results_dir "${OUT_DIR}" \
    --out_dir "${OUT_DIR}" \
    --split test

echo "Done."
