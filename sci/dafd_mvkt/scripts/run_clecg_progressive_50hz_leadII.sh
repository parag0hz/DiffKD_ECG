#!/bin/bash
# CLECG + Progressive HST-KD: Lead II 50Hz student via TA100.
#
# Steps:
#   1. Train TA100-II (from TA500-II), optionally with CLECG init
#   2. Train 50Hz student from TA100-II, optionally with CLECG init
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_clecg_progressive_50hz_leadII.sh
#
# Environment variables:
#   DATA_DIR                path to PTB-XL root
#   TEACHER_CKPT            teacher checkpoint
#   TA500_CKPT              TA-II-500Hz checkpoint
#   CLECG_ENCODER_100HZ     CLECG encoder (Lead II 100Hz) for TA100/student init
#   SEED                    random seed (default: 0)
#   BATCH_SIZE              batch size (default: 256)
#   OUT_DIR                 output directory (default: dafd_mvkt/outputs)

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
TA500_CKPT=${TA500_CKPT:-"dafd_mvkt/outputs/ta_II_500hz_seed0_best.pt"}
CLECG_ENCODER_100HZ=${CLECG_ENCODER_100HZ:-"dafd_mvkt/outputs/clecg_II_100hz_seed0_encoder.pt"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}

echo "========================================================"
echo "CLECG Progressive 50Hz  Lead=II  seed=${SEED}"
echo "TA500=${TA500_CKPT}"
echo "CLECG_100HZ=${CLECG_ENCODER_100HZ}"
echo "========================================================"

# Check CLECG availability
if [ ! -f "${CLECG_ENCODER_100HZ}" ]; then
    echo "WARNING: CLECG encoder not found: ${CLECG_ENCODER_100HZ}"
    SKIP_CLECG=1
else
    SKIP_CLECG=0
fi

# ── Step 1: Train TA100-II ─────────────────────────────────────────────────
TA100_NAME="ta_II_100hz_from_ta500_seed${SEED}"
TA100_CKPT="${OUT_DIR}/${TA100_NAME}_best.pt"

echo ""
echo "  Step 1: TA100-II (no CLECG init)"
if [ -f "${TA100_CKPT}" ]; then
    echo "  Checkpoint exists — skipping."
else
    python dafd_mvkt/train_temporal_ta.py \
        --config dafd_mvkt/configs/ta_ii_100hz_progressive.yaml \
        --data_dir "${DATA_DIR}" \
        --high_ta_ckpt "${TA500_CKPT}" \
        --lead "II" \
        --seed "${SEED}" \
        --batch_size "${BATCH_SIZE}" \
        --output_dir "${OUT_DIR}"
fi

# Evaluate TA100
TA100_METRICS="${OUT_DIR}/metrics_test_${TA100_NAME}.json"
if [ ! -f "${TA100_METRICS}" ]; then
    python dafd_mvkt/evaluate.py \
        --config dafd_mvkt/configs/ta_ii_100hz_progressive.yaml \
        --data_dir "${DATA_DIR}" \
        --ckpt "${TA100_CKPT}" \
        --split test \
        --model_name "${TA100_NAME}" \
        --output_dir "${OUT_DIR}" \
        --tune_thresholds
fi

# TA100 with CLECG init
if [ "${SKIP_CLECG}" = "0" ]; then
    TA100_CLECG_NAME="ta_II_100hz_from_ta500_clecg_seed${SEED}"
    TA100_CLECG_CKPT="${OUT_DIR}/${TA100_CLECG_NAME}_best.pt"
    echo ""
    echo "  Step 1b: TA100-II (CLECG init)"
    if [ ! -f "${TA100_CLECG_CKPT}" ]; then
        python dafd_mvkt/train_temporal_ta.py \
            --config dafd_mvkt/configs/ta_ii_100hz_progressive.yaml \
            --data_dir "${DATA_DIR}" \
            --high_ta_ckpt "${TA500_CKPT}" \
            --lead "II" \
            --seed "${SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --run_name "${TA100_CLECG_NAME}" \
            --init_encoder_ckpt "${CLECG_ENCODER_100HZ}" \
            --output_dir "${OUT_DIR}"
    fi
fi

# ── Step 2: 50Hz student from TA100 ───────────────────────────────────────
run_student() {
    local LOSSES=$1
    local TA_CKPT_ARG=$2
    local SUFFIX=$3
    local INIT_ENC=$4

    local LOSSES_FNAME
    LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
    local MODEL_NAME="student_II_50hz_${LOSSES_FNAME}${SUFFIX}_seed${SEED}"
    local CKPT="${OUT_DIR}/${MODEL_NAME}_best.pt"
    local METRICS="${OUT_DIR}/metrics_test_${MODEL_NAME}.json"

    echo ""
    echo "  Train: ${MODEL_NAME}"

    local EXTRA_ARGS=""
    if [ -n "${INIT_ENC}" ] && [ -f "${INIT_ENC}" ]; then
        EXTRA_ARGS="--init_encoder_ckpt ${INIT_ENC}"
    fi

    if [ -f "${CKPT}" ]; then
        echo "  Checkpoint exists — skipping."
    else
        python dafd_mvkt/train_student_hier.py \
            --config dafd_mvkt/configs/student_50hz_progressive.yaml \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --ta_ckpt "${TA_CKPT_ARG}" \
            --ta_config dafd_mvkt/configs/ta_ii_100hz_progressive.yaml \
            --lead "II" \
            --losses "${LOSSES}" \
            --ta_hz 100 \
            --seed "${SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --run_name "${MODEL_NAME}" \
            --output_dir "${OUT_DIR}" \
            ${EXTRA_ARGS}
    fi

    if [ ! -f "${METRICS}" ]; then
        python dafd_mvkt/evaluate.py \
            --config dafd_mvkt/configs/student_50hz_progressive.yaml \
            --data_dir "${DATA_DIR}" \
            --ckpt "${CKPT}" \
            --split test \
            --model_name "${MODEL_NAME}" \
            --output_dir "${OUT_DIR}" \
            --tune_thresholds
    fi
}

echo ""
echo "══ Step 2: 50Hz students from TA100 ══"

# Standard progressive (no CLECG)
run_student "bce,ta_mkd,ta_crf,feature" "${TA100_CKPT}" "_prog" ""

if [ "${SKIP_CLECG}" = "0" ]; then
    # CLECG student init + standard TA100
    run_student "bce,ta_mkd,ta_crf,feature" "${TA100_CKPT}" "_prog_clecg" "${CLECG_ENCODER_100HZ}"

    # CLECG TA100 + CLECG student
    if [ -f "${TA100_CLECG_CKPT}" ]; then
        run_student "bce,ta_mkd,ta_crf,feature" "${TA100_CLECG_CKPT}" "_prog_clecgta_clecg" "${CLECG_ENCODER_100HZ}"
    fi
fi

echo ""
echo "CLECG Progressive 50Hz Lead II complete."
