#!/bin/bash
# Auto Stage 4: 50Hz trade-off experiments (Lead I and Lead II).
#
# For each lead runs:
#   a. BCE baseline (50Hz student, direct TA500 distillation)
#   b. Full HST-KD (50Hz student)
#   c. Full HST-KD + CLECG student encoder init (if encoder available)
#
# Results go to OUT_DIR (dafd_mvkt/outputs) and are picked up by
# auto_collect_key_results.py.
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl SEED=0 bash dafd_mvkt/scripts/auto_stage_50hz_tradeoff.sh

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
CLECG_DIR="${OUT_DIR}/clecg"

echo "========================================================"
echo "AUTO STAGE: 50Hz Trade-off  seed=${SEED}"
echo "DATA_DIR=${DATA_DIR}  OUT_DIR=${OUT_DIR}"
echo "========================================================"

_run_50hz_student() {
    local LEAD="$1" LOSSES="$2" SUFFIX="$3" INIT_ENC="$4"
    local LOSSES_FNAME
    LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
    local NAME="student_${LEAD}_50hz_${LOSSES_FNAME}${SUFFIX}_seed${SEED}"
    local CKPT="${OUT_DIR}/${NAME}_best.pt"
    local METRICS="${OUT_DIR}/metrics_test_${NAME}.json"

    local TA_CKPT="${OUT_DIR}/ta_${LEAD}_500hz_seed${SEED}_best.pt"
    if [ ! -f "${TA_CKPT}" ]; then
        echo "  WARNING: TA-${LEAD}-500Hz not found: ${TA_CKPT} — skipping ${NAME}"
        return 0
    fi

    # Select config based on lead
    if [ "${LEAD}" = "I" ]; then
        STUDENT_CFG="dafd_mvkt/configs/student_50hz_ta.yaml"
        TA_CFG_ARG="--ta_config dafd_mvkt/configs/ta_i_500hz.yaml"
        EVAL_CFG="dafd_mvkt/configs/student_50hz_ta.yaml"
    else
        STUDENT_CFG="dafd_mvkt/configs/student_50hz_ta.yaml"
        TA_CFG_ARG=""
        EVAL_CFG="dafd_mvkt/configs/student_50hz_ta.yaml"
    fi

    echo ""
    echo "  ── ${NAME}"

    local EXTRA=""
    [ -n "${INIT_ENC}" ] && [ -f "${INIT_ENC}" ] && EXTRA="--init_encoder_ckpt ${INIT_ENC}"

    if [ ! -f "${CKPT}" ]; then
        python dafd_mvkt/train_student_hier.py \
            --config "${STUDENT_CFG}" \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --ta_ckpt "${TA_CKPT}" \
            ${TA_CFG_ARG} \
            --lead "${LEAD}" \
            --losses "${LOSSES}" \
            --seed "${SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --run_name "${NAME}" \
            --output_dir "${OUT_DIR}" \
            ${EXTRA}
    else
        echo "  Checkpoint exists — skipping training."
    fi

    if [ ! -f "${METRICS}" ]; then
        python dafd_mvkt/evaluate.py \
            --config "${EVAL_CFG}" \
            --data_dir "${DATA_DIR}" \
            --ckpt "${CKPT}" \
            --split test \
            --model_name "${NAME}" \
            --output_dir "${OUT_DIR}" \
            --tune_thresholds
    else
        echo "  Metrics exist — skipping evaluation."
    fi
}

# ── Lead I 50Hz ───────────────────────────────────────────────────────────────
echo ""
echo "=== Lead I 50Hz ==="
ENC_I_100="${CLECG_DIR}/clecg_I_100hz_seed${SEED}_encoder.pt"

_run_50hz_student "I" "bce"                        ""       ""
_run_50hz_student "I" "bce,ta_mkd,ta_crf,feature"  ""       ""
if [ -f "${ENC_I_100}" ]; then
    _run_50hz_student "I" "bce,ta_mkd,ta_crf,feature"  "_clecg"  "${ENC_I_100}"
fi

# ── Lead II 50Hz ──────────────────────────────────────────────────────────────
echo ""
echo "=== Lead II 50Hz ==="
ENC_II_100="${CLECG_DIR}/clecg_II_100hz_seed${SEED}_encoder.pt"

_run_50hz_student "II" "bce"                        ""       ""
_run_50hz_student "II" "bce,ta_mkd,ta_crf,feature"  ""       ""
if [ -f "${ENC_II_100}" ]; then
    _run_50hz_student "II" "bce,ta_mkd,ta_crf,feature"  "_clecg"  "${ENC_II_100}"
fi

echo ""
echo "AUTO STAGE 50Hz trade-off complete."
