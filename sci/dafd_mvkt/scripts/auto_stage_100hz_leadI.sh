#!/bin/bash
# Auto Stage 1: Lead I 100Hz complete experiment ladder.
#
# Steps (sequential, maximum VRAM usage per run):
#   1. Train standard TA-I-500Hz (if not present)
#   2. CLECG pretrain 500Hz Lead I (for TA encoder init)
#   3. CLECG pretrain 100Hz Lead I (for student encoder init)
#   4. Train CLECG-strengthened TA-I-500Hz
#   5. Student variants:
#      a. BCE baseline
#      b. BCE + ta_mkd (partial KD)
#      c. Full HST-KD (bce,ta_mkd,ta_crf,feature)
#      d. Full HST-KD + CLECG student encoder init
#      e. Full HST-KD + CLECG TA + CLECG student encoder (full CLECG pipeline)
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl SEED=0 bash dafd_mvkt/scripts/auto_stage_100hz_leadI.sh
#
# Environment variables:
#   DATA_DIR       PTB-XL root
#   TEACHER_CKPT   teacher checkpoint (default: dafd_mvkt/outputs/teacher_best.pt)
#   SEED           random seed (default: 0)
#   BATCH_SIZE     batch size (default: 256)
#   OUT_DIR        output directory (default: dafd_mvkt/outputs)

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
CLECG_DIR="${OUT_DIR}/clecg"

TA_STD="${OUT_DIR}/ta_I_500hz_seed${SEED}_best.pt"
TA_CLECG="${OUT_DIR}/ta_I_500hz_clecg_seed${SEED}_best.pt"
ENC_500="${CLECG_DIR}/clecg_I_500hz_seed${SEED}_encoder.pt"
ENC_100="${CLECG_DIR}/clecg_I_100hz_seed${SEED}_encoder.pt"

mkdir -p "${CLECG_DIR}"

echo "========================================================"
echo "AUTO STAGE: Lead I 100Hz  seed=${SEED}"
echo "DATA_DIR=${DATA_DIR}  OUT_DIR=${OUT_DIR}"
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

# ── 1. Standard TA-I-500Hz ────────────────────────────────────────────────────
echo ""
echo "[1/5] TA-I-500Hz (standard)"
if [ ! -f "${TA_STD}" ]; then
    python dafd_mvkt/train_ta.py \
        --config dafd_mvkt/configs/ta_i_500hz.yaml \
        --data_dir "${DATA_DIR}" \
        --teacher_ckpt "${TEACHER_CKPT}" \
        --lead I \
        --batch_size "${BATCH_SIZE}" \
        --seed "${SEED}" \
        --run_name "ta_I_500hz_seed${SEED}" \
        --output_dir "${OUT_DIR}"
else
    echo "  Checkpoint exists: ${TA_STD}"
fi
TA_STD_METRICS="${OUT_DIR}/metrics_test_ta_I_500hz_seed${SEED}.json"
if [ -f "${TA_STD}" ] && [ ! -f "${TA_STD_METRICS}" ]; then
    python dafd_mvkt/evaluate.py \
        --config dafd_mvkt/configs/ta_i_500hz.yaml \
        --data_dir "${DATA_DIR}" \
        --ckpt "${TA_STD}" \
        --split test \
        --model_name "ta_I_500hz_seed${SEED}" \
        --output_dir "${OUT_DIR}" \
        --tune_thresholds
fi

# ── 2. CLECG pretraining 500Hz (for TA init) ─────────────────────────────────
echo ""
echo "[2/5] CLECG pretraining Lead I 500Hz (for TA init)"
if [ ! -f "${ENC_500}" ]; then
    python dafd_mvkt/pretrain_clecg.py \
        --config dafd_mvkt/configs/clecg_leadI_500hz.yaml \
        --data_dir "${DATA_DIR}" \
        --lead I \
        --hz 500 \
        --seed "${SEED}" \
        --batch_size "${BATCH_SIZE}" \
        --run_name "clecg_I_500hz_seed${SEED}" \
        --output_dir "${CLECG_DIR}"
else
    echo "  Encoder exists: ${ENC_500}"
fi

# ── 3. CLECG pretraining 100Hz (for student init) ────────────────────────────
echo ""
echo "[3/5] CLECG pretraining Lead I 100Hz (for student init)"
if [ ! -f "${ENC_100}" ]; then
    python dafd_mvkt/pretrain_clecg.py \
        --config dafd_mvkt/configs/clecg_leadI_100hz.yaml \
        --data_dir "${DATA_DIR}" \
        --lead I \
        --hz 100 \
        --seed "${SEED}" \
        --batch_size "${BATCH_SIZE}" \
        --run_name "clecg_I_100hz_seed${SEED}" \
        --output_dir "${CLECG_DIR}"
else
    echo "  Encoder exists: ${ENC_100}"
fi

# ── 4. CLECG-strengthened TA-I-500Hz ─────────────────────────────────────────
echo ""
echo "[4/5] TA-I-500Hz with CLECG encoder init"
if [ ! -f "${ENC_500}" ]; then
    echo "  WARNING: 500Hz CLECG encoder missing — skipping CLECG-TA."
elif [ ! -f "${TA_CLECG}" ]; then
    python dafd_mvkt/train_ta.py \
        --config dafd_mvkt/configs/ta_i_500hz.yaml \
        --data_dir "${DATA_DIR}" \
        --teacher_ckpt "${TEACHER_CKPT}" \
        --lead I \
        --batch_size "${BATCH_SIZE}" \
        --seed "${SEED}" \
        --run_name "ta_I_500hz_clecg_seed${SEED}" \
        --output_dir "${OUT_DIR}" \
        --init_encoder_ckpt "${ENC_500}"
else
    echo "  Checkpoint exists: ${TA_CLECG}"
fi
TA_CLECG_METRICS="${OUT_DIR}/metrics_test_ta_I_500hz_clecg_seed${SEED}.json"
if [ -f "${TA_CLECG}" ] && [ ! -f "${TA_CLECG_METRICS}" ]; then
    python dafd_mvkt/evaluate.py \
        --config dafd_mvkt/configs/ta_i_500hz.yaml \
        --data_dir "${DATA_DIR}" \
        --ckpt "${TA_CLECG}" \
        --split test \
        --model_name "ta_I_500hz_clecg_seed${SEED}" \
        --output_dir "${OUT_DIR}" \
        --tune_thresholds
fi

# ── 5. Student experiments ────────────────────────────────────────────────────
echo ""
echo "[5/5] Student experiments"

# 5a: Standard TA — ablation ladder
_run_student "bce"                        ""              ""          "${TA_STD}"
_run_student "bce,ta_mkd"                 ""              ""          "${TA_STD}"
_run_student "bce,ta_mkd,ta_crf,feature"  ""              ""          "${TA_STD}"

# 5b: Full HST-KD + CLECG student encoder
if [ -f "${ENC_100}" ]; then
    _run_student "bce,ta_mkd,ta_crf,feature"  "_clecg"          "${ENC_100}"  "${TA_STD}"
fi

# 5c: Full HST-KD + CLECG TA + CLECG student encoder (full CLECG pipeline)
if [ -f "${TA_CLECG}" ] && [ -f "${ENC_100}" ]; then
    _run_student "bce,ta_mkd,ta_crf,feature"  "_clecgta_clecg"  "${ENC_100}"  "${TA_CLECG}"
fi

echo ""
echo "AUTO STAGE Lead I 100Hz complete."
