#!/bin/bash
# SimCLR + Progressive HST-KD: Lead II 50Hz trade-off experiment.
#
# Runs:
#   1. SimCLR pretraining Lead II 50Hz  (for student init)
#   2. 50Hz BCE baseline (should already exist)
#   3. 50Hz Progressive HST-KD (should already exist)
#   4. 50Hz Progressive HST-KD + SimCLR student encoder
#   5. (optional) 50Hz Progressive HST-KD + SimCLR TA100 + SimCLR student
#
# Comparison context (50Hz Lead II):
#   BCE 50Hz:               AUC 0.8061 / F1 0.5740
#   Progressive HST-KD:     AUC 0.8320 / F1 0.6106
#   Target: beat AUC 0.8320 and/or F1 0.6106
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl SEED=0 \
#   bash dafd_mvkt/scripts/run_simclr_progressive_50hz_leadII.sh

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
SIMCLR_DIR="${OUT_DIR}/simclr"

TA500_STD="${OUT_DIR}/ta_II_500hz_seed${SEED}_best.pt"
ENC_50="${SIMCLR_DIR}/simclr_II_50hz_seed${SEED}_encoder.pt"
# TA100 (progressive): may use existing progressive TA
TA100_PROG="${OUT_DIR}/ta_II_100hz_from_ta500_seed${SEED}_best.pt"

mkdir -p "${SIMCLR_DIR}"

echo "========================================================"
echo "SimCLR + Progressive 50Hz  Lead=II  seed=${SEED}"
echo "========================================================"

_run_50hz() {
    local LOSSES="$1" SUFFIX="$2" INIT_ENC="$3"
    local LOSSES_FNAME
    LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
    local NAME="student_II_50hz_${LOSSES_FNAME}${SUFFIX}_seed${SEED}"
    local CKPT="${OUT_DIR}/${NAME}_best.pt"
    local METRICS="${OUT_DIR}/metrics_test_${NAME}.json"

    echo ""
    echo "  ── ${NAME}"

    if [ ! -f "${TA500_STD}" ]; then
        echo "  ERROR: TA-II-500Hz not found: ${TA500_STD}"
        return 1
    fi

    # Select TA for progressive: prefer TA100 progressive if available
    local TA_FOR_PROG="${TA500_STD}"
    local PROG_CFG=""
    if [ -f "${TA100_PROG}" ]; then
        TA_FOR_PROG="${TA100_PROG}"
        PROG_CFG="--config dafd_mvkt/configs/student_50hz_progressive.yaml"
    fi

    local EXTRA=""
    [ -n "${INIT_ENC}" ] && [ -f "${INIT_ENC}" ] && EXTRA="--init_encoder_ckpt ${INIT_ENC}"

    if [ ! -f "${CKPT}" ]; then
        python dafd_mvkt/train_student_hier.py \
            ${PROG_CFG:---config dafd_mvkt/configs/student_50hz_ta.yaml} \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --ta_ckpt "${TA_FOR_PROG}" \
            --lead II \
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
        local EVAL_CFG="dafd_mvkt/configs/student_50hz_ta.yaml"
        [ -f "${TA100_PROG}" ] && EVAL_CFG="dafd_mvkt/configs/student_50hz_progressive.yaml"
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

# ── 1. SimCLR pretraining 50Hz ────────────────────────────────────────────────
echo ""
echo "[1/3] SimCLR pretraining Lead II 50Hz"
if [ ! -f "${ENC_50}" ]; then
    python dafd_mvkt/pretrain_simclr.py \
        --config dafd_mvkt/configs/simclr_leadII_50hz.yaml \
        --data_dir "${DATA_DIR}" \
        --lead II \
        --hz 50 \
        --seed "${SEED}" \
        --batch_size "${BATCH_SIZE}" \
        --run_name "simclr_II_50hz_seed${SEED}" \
        --output_dir "${SIMCLR_DIR}"
else
    echo "  Encoder exists: ${ENC_50}"
fi

# ── 2. 50Hz Progressive HST-KD reference (no pretrain) ───────────────────────
echo ""
echo "[2/3] 50Hz Progressive HST-KD reference runs"
_run_50hz "bce"                        ""       ""
_run_50hz "bce,ta_mkd,ta_crf,feature"  "_prog"  ""

# ── 3. 50Hz Progressive HST-KD + SimCLR student encoder ─────────────────────
echo ""
echo "[3/3] 50Hz Progressive HST-KD + SimCLR student encoder"
if [ -f "${ENC_50}" ]; then
    _run_50hz "bce,ta_mkd,ta_crf,feature"  "_prog_simclr"  "${ENC_50}"
else
    echo "  WARNING: 50Hz SimCLR encoder missing — skipping."
fi

echo ""
echo "SimCLR 50Hz progressive experiment complete."
echo "Results in ${OUT_DIR}"
