#!/bin/bash
# SimCLR + HST-KD experiments: Lead I 100Hz student.
#
# Runs:
#   1. SimCLR pretraining Lead I 100Hz (if encoder missing)
#   2. SimCLR pretraining Lead I 500Hz (for TA init, if RUN_TA_SIMCLR=1)
#   3. SimCLR-initialized TA-I-500Hz   (if RUN_TA_SIMCLR=1)
#   4. HST-KD student + SimCLR student init
#   5. HST-KD student + SimCLR TA + SimCLR student init
#
# Comparison context:
#   Random HST-KD (baseline):           AUC 0.8292 / F1 0.6064
#   CLECG HST-KD (current best Lead I): AUC 0.8349 / F1 0.6072
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl SEED=0 RUN_TA_SIMCLR=1 \
#   bash dafd_mvkt/scripts/run_simclr_hst_100hz_leadI.sh

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
SIMCLR_DIR="${OUT_DIR}/simclr"
RUN_TA_SIMCLR=${RUN_TA_SIMCLR:-1}

TA_STD="${OUT_DIR}/ta_I_500hz_seed${SEED}_best.pt"
TA_SIMCLR="${OUT_DIR}/ta_I_500hz_simclr_seed${SEED}_best.pt"
ENC_100="${SIMCLR_DIR}/simclr_I_100hz_seed${SEED}_encoder.pt"
ENC_500="${SIMCLR_DIR}/simclr_I_500hz_seed${SEED}_encoder.pt"

mkdir -p "${SIMCLR_DIR}"

echo "========================================================"
echo "SimCLR + HST-KD  Lead=I  100Hz  seed=${SEED}"
echo "DATA_DIR=${DATA_DIR}  RUN_TA_SIMCLR=${RUN_TA_SIMCLR}"
echo "========================================================"

_run_student() {
    local LOSSES="$1" SUFFIX="$2" INIT_ENC="$3" TA_CKPT_USE="$4"
    local LOSSES_FNAME
    LOSSES_FNAME=$(echo "$LOSSES" | tr ',' '_')
    local NAME="student_I_100hz_${LOSSES_FNAME}${SUFFIX}_seed${SEED}"
    local CKPT="${OUT_DIR}/${NAME}_best.pt"
    local METRICS="${OUT_DIR}/metrics_test_${NAME}.json"

    echo ""
    echo "  ── ${NAME}"

    local EXTRA=""
    [ -n "${INIT_ENC}" ] && [ -f "${INIT_ENC}" ] && EXTRA="--init_encoder_ckpt ${INIT_ENC}"

    if [ ! -f "${CKPT}" ]; then
        python dafd_mvkt/train_student_hier.py \
            --config dafd_mvkt/configs/student_100hz_leadI.yaml \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --ta_ckpt "${TA_CKPT_USE}" \
            --ta_config dafd_mvkt/configs/ta_i_500hz.yaml \
            --lead I \
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
            --config dafd_mvkt/configs/student_100hz_leadI.yaml \
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

# ── 1. SimCLR pretraining 100Hz ───────────────────────────────────────────────
echo ""
echo "[1/4] SimCLR pretraining Lead I 100Hz"
if [ ! -f "${ENC_100}" ]; then
    python dafd_mvkt/pretrain_simclr.py \
        --config dafd_mvkt/configs/simclr_leadI_100hz.yaml \
        --data_dir "${DATA_DIR}" \
        --lead I \
        --hz 100 \
        --seed "${SEED}" \
        --batch_size "${BATCH_SIZE}" \
        --run_name "simclr_I_100hz_seed${SEED}" \
        --output_dir "${SIMCLR_DIR}"
else
    echo "  Encoder exists: ${ENC_100}"
fi

# ── 2-3. SimCLR TA (optional) ─────────────────────────────────────────────────
if [ "${RUN_TA_SIMCLR}" = "1" ]; then
    echo ""
    echo "[2/4] SimCLR pretraining Lead I 500Hz (for TA init)"
    if [ ! -f "${ENC_500}" ]; then
        python dafd_mvkt/pretrain_simclr.py \
            --config dafd_mvkt/configs/simclr_leadI_500hz.yaml \
            --data_dir "${DATA_DIR}" \
            --lead I \
            --hz 500 \
            --seed "${SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --run_name "simclr_I_500hz_seed${SEED}" \
            --output_dir "${SIMCLR_DIR}"
    else
        echo "  Encoder exists: ${ENC_500}"
    fi

    echo ""
    echo "[3/4] TA-I-500Hz with SimCLR encoder init"
    if [ ! -f "${TA_STD}" ]; then
        echo "  Training standard TA-I-500Hz first..."
        python dafd_mvkt/train_ta.py \
            --config dafd_mvkt/configs/ta_i_500hz.yaml \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --lead I \
            --batch_size "${BATCH_SIZE}" \
            --seed "${SEED}" \
            --run_name "ta_I_500hz_seed${SEED}" \
            --output_dir "${OUT_DIR}"
    fi
    if [ -f "${ENC_500}" ] && [ ! -f "${TA_SIMCLR}" ]; then
        python dafd_mvkt/train_ta.py \
            --config dafd_mvkt/configs/ta_i_500hz.yaml \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --lead I \
            --batch_size "${BATCH_SIZE}" \
            --seed "${SEED}" \
            --run_name "ta_I_500hz_simclr_seed${SEED}" \
            --output_dir "${OUT_DIR}" \
            --init_encoder_ckpt "${ENC_500}"
    elif [ -f "${TA_SIMCLR}" ]; then
        echo "  SimCLR TA checkpoint exists: ${TA_SIMCLR}"
    fi
    TA_SIMCLR_METRICS="${OUT_DIR}/metrics_test_ta_I_500hz_simclr_seed${SEED}.json"
    if [ -f "${TA_SIMCLR}" ] && [ ! -f "${TA_SIMCLR_METRICS}" ]; then
        python dafd_mvkt/evaluate.py \
            --config dafd_mvkt/configs/ta_i_500hz.yaml \
            --data_dir "${DATA_DIR}" \
            --ckpt "${TA_SIMCLR}" \
            --split test \
            --model_name "ta_I_500hz_simclr_seed${SEED}" \
            --output_dir "${OUT_DIR}" \
            --tune_thresholds
    fi
else
    echo "[2-3/4] Skipped (RUN_TA_SIMCLR=0)"
fi

# ── 4. Student experiments ────────────────────────────────────────────────────
echo ""
echo "[4/4] Student experiments"

if [ ! -f "${TA_STD}" ]; then
    echo "  ERROR: standard TA not found: ${TA_STD}"
    echo "  Run auto_stage_100hz_leadI.sh first."
    exit 1
fi

if [ -f "${ENC_100}" ]; then
    _run_student "bce,ta_mkd,ta_crf,feature"  "_simclr"          "${ENC_100}"  "${TA_STD}"
fi

if [ "${RUN_TA_SIMCLR}" = "1" ] && [ -f "${TA_SIMCLR}" ] && [ -f "${ENC_100}" ]; then
    _run_student "bce,ta_mkd,ta_crf,feature"  "_simclrta_simclr" "${ENC_100}"  "${TA_SIMCLR}"
fi

echo ""
echo "SimCLR+HST-KD Lead I 100Hz complete."
echo "Results in ${OUT_DIR}"
