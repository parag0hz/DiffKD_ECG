#!/bin/bash
# Auto Stage 5: Multi-seed validation for the best-performing method.
#
# Reads BEST_LEAD_100HZ from outputs/auto/decision.env.
# Runs seeds 1 and 2 for the full CLECG pipeline on the best lead
# (seed 0 already done by earlier stages).
#
# Also runs TA seeds 1,2 since TA is needed for each student seed.
# CLECG pretraining seeds 1,2 are run too (needed for student encoder init).
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/auto_stage_multiseed.sh

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
SEEDS=${SEEDS:-"1 2"}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
CLECG_DIR="${OUT_DIR}/clecg"
AUTO_DIR="${OUT_DIR}/auto"

# Load best lead from decider output
DECISION_ENV="${AUTO_DIR}/decision.env"
if [ -f "${DECISION_ENV}" ]; then
    source "${DECISION_ENV}"
fi
LEAD=${BEST_LEAD_100HZ:-"I"}

echo "========================================================"
echo "AUTO STAGE: Multi-seed  lead=${LEAD}  seeds=${SEEDS}"
echo "========================================================"

# Select configs based on lead
if [ "${LEAD}" = "I" ]; then
    TA_CFG="dafd_mvkt/configs/ta_i_500hz.yaml"
    STUDENT_CFG="dafd_mvkt/configs/student_100hz_leadI.yaml"
    TA_CFG_ARG="--ta_config dafd_mvkt/configs/ta_i_500hz.yaml"
    CLECG_500_CFG="dafd_mvkt/configs/clecg_leadI_500hz.yaml"
    CLECG_100_CFG="dafd_mvkt/configs/clecg_leadI_100hz.yaml"
else
    TA_CFG="dafd_mvkt/configs/ta_ii_500hz.yaml"
    STUDENT_CFG="dafd_mvkt/configs/student_100hz_ta.yaml"
    TA_CFG_ARG=""
    CLECG_500_CFG="dafd_mvkt/configs/clecg_leadII_500hz.yaml"
    CLECG_100_CFG="dafd_mvkt/configs/clecg_leadII_100hz.yaml"
fi

LOSSES="bce,ta_mkd,ta_crf,feature"
LOSSES_FNAME=$(echo "${LOSSES}" | tr ',' '_')

for SEED in ${SEEDS}; do
    echo ""
    echo "══════════════  seed=${SEED}  ══════════════"

    TA_STD="${OUT_DIR}/ta_${LEAD}_500hz_seed${SEED}_best.pt"
    TA_CLECG="${OUT_DIR}/ta_${LEAD}_500hz_clecg_seed${SEED}_best.pt"
    ENC_500="${CLECG_DIR}/clecg_${LEAD}_500hz_seed${SEED}_encoder.pt"
    ENC_100="${CLECG_DIR}/clecg_${LEAD}_100hz_seed${SEED}_encoder.pt"

    # ── TA ──────────────────────────────────────────────────────────────────
    if [ ! -f "${TA_STD}" ]; then
        echo "  Training TA-${LEAD}-500Hz seed=${SEED}..."
        python dafd_mvkt/train_ta.py \
            --config "${TA_CFG}" \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --lead "${LEAD}" \
            --batch_size "${BATCH_SIZE}" \
            --seed "${SEED}" \
            --run_name "ta_${LEAD}_500hz_seed${SEED}" \
            --output_dir "${OUT_DIR}"
    fi
    TA_METRICS="${OUT_DIR}/metrics_test_ta_${LEAD}_500hz_seed${SEED}.json"
    if [ -f "${TA_STD}" ] && [ ! -f "${TA_METRICS}" ]; then
        python dafd_mvkt/evaluate.py \
            --config "${TA_CFG}" \
            --data_dir "${DATA_DIR}" \
            --ckpt "${TA_STD}" \
            --split test \
            --model_name "ta_${LEAD}_500hz_seed${SEED}" \
            --output_dir "${OUT_DIR}" \
            --tune_thresholds
    fi

    # ── CLECG 500Hz (for TA init) ────────────────────────────────────────────
    if [ ! -f "${ENC_500}" ]; then
        echo "  CLECG pretraining ${LEAD} 500Hz seed=${SEED}..."
        python dafd_mvkt/pretrain_clecg.py \
            --config "${CLECG_500_CFG}" \
            --data_dir "${DATA_DIR}" \
            --lead "${LEAD}" \
            --hz 500 \
            --seed "${SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --run_name "clecg_${LEAD}_500hz_seed${SEED}" \
            --output_dir "${CLECG_DIR}"
    fi

    # ── CLECG TA ─────────────────────────────────────────────────────────────
    if [ -f "${ENC_500}" ] && [ ! -f "${TA_CLECG}" ]; then
        echo "  Training CLECG TA-${LEAD}-500Hz seed=${SEED}..."
        python dafd_mvkt/train_ta.py \
            --config "${TA_CFG}" \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --lead "${LEAD}" \
            --batch_size "${BATCH_SIZE}" \
            --seed "${SEED}" \
            --run_name "ta_${LEAD}_500hz_clecg_seed${SEED}" \
            --output_dir "${OUT_DIR}" \
            --init_encoder_ckpt "${ENC_500}"
    fi
    TA_CLECG_METRICS="${OUT_DIR}/metrics_test_ta_${LEAD}_500hz_clecg_seed${SEED}.json"
    if [ -f "${TA_CLECG}" ] && [ ! -f "${TA_CLECG_METRICS}" ]; then
        python dafd_mvkt/evaluate.py \
            --config "${TA_CFG}" \
            --data_dir "${DATA_DIR}" \
            --ckpt "${TA_CLECG}" \
            --split test \
            --model_name "ta_${LEAD}_500hz_clecg_seed${SEED}" \
            --output_dir "${OUT_DIR}" \
            --tune_thresholds
    fi

    # ── CLECG 100Hz (for student init) ──────────────────────────────────────
    if [ ! -f "${ENC_100}" ]; then
        echo "  CLECG pretraining ${LEAD} 100Hz seed=${SEED}..."
        python dafd_mvkt/pretrain_clecg.py \
            --config "${CLECG_100_CFG}" \
            --data_dir "${DATA_DIR}" \
            --lead "${LEAD}" \
            --hz 100 \
            --seed "${SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --run_name "clecg_${LEAD}_100hz_seed${SEED}" \
            --output_dir "${CLECG_DIR}"
    fi

    # ── Student (no CLECG) ───────────────────────────────────────────────────
    S_NAME="student_${LEAD}_100hz_${LOSSES_FNAME}_seed${SEED}"
    S_CKPT="${OUT_DIR}/${S_NAME}_best.pt"
    if [ ! -f "${S_CKPT}" ]; then
        echo "  Training student (no CLECG) seed=${SEED}..."
        python dafd_mvkt/train_student_hier.py \
            --config "${STUDENT_CFG}" \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --ta_ckpt "${TA_STD}" \
            ${TA_CFG_ARG} \
            --lead "${LEAD}" \
            --losses "${LOSSES}" \
            --seed "${SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --run_name "${S_NAME}" \
            --output_dir "${OUT_DIR}"
    fi
    S_METRICS="${OUT_DIR}/metrics_test_${S_NAME}.json"
    if [ -f "${S_CKPT}" ] && [ ! -f "${S_METRICS}" ]; then
        python dafd_mvkt/evaluate.py \
            --config "${STUDENT_CFG}" \
            --data_dir "${DATA_DIR}" \
            --ckpt "${S_CKPT}" \
            --split test \
            --model_name "${S_NAME}" \
            --output_dir "${OUT_DIR}" \
            --tune_thresholds
    fi

    # ── Student + CLECG (full pipeline) ─────────────────────────────────────
    if [ -f "${ENC_100}" ]; then
        BEST_TA_FOR_STUDENT="${TA_CLECG}"
        [ ! -f "${BEST_TA_FOR_STUDENT}" ] && BEST_TA_FOR_STUDENT="${TA_STD}"

        if [ -f "${TA_CLECG}" ]; then
            SUFFIX="_clecgta_clecg"
        else
            SUFFIX="_clecg"
        fi

        SC_NAME="student_${LEAD}_100hz_${LOSSES_FNAME}${SUFFIX}_seed${SEED}"
        SC_CKPT="${OUT_DIR}/${SC_NAME}_best.pt"
        SC_METRICS="${OUT_DIR}/metrics_test_${SC_NAME}.json"

        if [ ! -f "${SC_CKPT}" ]; then
            echo "  Training student (CLECG) seed=${SEED}..."
            python dafd_mvkt/train_student_hier.py \
                --config "${STUDENT_CFG}" \
                --data_dir "${DATA_DIR}" \
                --teacher_ckpt "${TEACHER_CKPT}" \
                --ta_ckpt "${BEST_TA_FOR_STUDENT}" \
                ${TA_CFG_ARG} \
                --lead "${LEAD}" \
                --losses "${LOSSES}" \
                --seed "${SEED}" \
                --batch_size "${BATCH_SIZE}" \
                --run_name "${SC_NAME}" \
                --output_dir "${OUT_DIR}" \
                --init_encoder_ckpt "${ENC_100}"
        fi
        if [ -f "${SC_CKPT}" ] && [ ! -f "${SC_METRICS}" ]; then
            python dafd_mvkt/evaluate.py \
                --config "${STUDENT_CFG}" \
                --data_dir "${DATA_DIR}" \
                --ckpt "${SC_CKPT}" \
                --split test \
                --model_name "${SC_NAME}" \
                --output_dir "${OUT_DIR}" \
                --tune_thresholds
        fi
    fi
done

echo ""
echo "AUTO STAGE Multi-seed complete."
echo "Aggregating results..."
python dafd_mvkt/experiments/aggregate_results.py \
    --results_dir "${OUT_DIR}" \
    --out_dir "${OUT_DIR}" \
    --split test
echo "Done."
