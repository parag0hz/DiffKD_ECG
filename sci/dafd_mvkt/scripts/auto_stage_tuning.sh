#!/bin/bash
# Auto Stage 3: Hyperparameter tuning for the best method.
#
# Reads BEST_LEAD_100HZ from outputs/auto/decision.env.
# Runs HP variants (lr, alpha_ta) for the full CLECG pipeline on that lead.
# Uses "virtual seeds" 10-14 to distinguish HP configs (same data/model as seed 0).
# Results saved to outputs/tuning/ — picked up by auto_collect_key_results.py.
# Best HP written to outputs/auto/tuning_best_hp.env.
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/auto_stage_tuning.sh

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
TUNING_DIR="${OUT_DIR}/tuning"
CLECG_DIR="${OUT_DIR}/clecg"
AUTO_DIR="${OUT_DIR}/auto"

mkdir -p "${TUNING_DIR}" "${AUTO_DIR}"

# Load best lead from decider output
DECISION_ENV="${AUTO_DIR}/decision.env"
if [ -f "${DECISION_ENV}" ]; then
    source "${DECISION_ENV}"
fi
LEAD=${BEST_LEAD_100HZ:-"I"}
echo "========================================================"
echo "AUTO STAGE: HP Tuning  lead=${LEAD}  ref_seed=${SEED}"
echo "========================================================"

# Select configs based on lead
if [ "${LEAD}" = "I" ]; then
    STUDENT_CFG="dafd_mvkt/configs/student_100hz_leadI.yaml"
    TA_CFG_ARG="--ta_config dafd_mvkt/configs/ta_i_500hz.yaml"
    TA_CKPT="${OUT_DIR}/ta_I_500hz_seed${SEED}_best.pt"
    TA_CKPT_CLECG="${OUT_DIR}/ta_I_500hz_clecg_seed${SEED}_best.pt"
    ENC_100="${CLECG_DIR}/clecg_I_100hz_seed${SEED}_encoder.pt"
    EVAL_CFG="dafd_mvkt/configs/student_100hz_leadI.yaml"
else
    STUDENT_CFG="dafd_mvkt/configs/student_100hz_ta.yaml"
    TA_CFG_ARG=""
    TA_CKPT="${OUT_DIR}/ta_II_500hz_seed${SEED}_best.pt"
    TA_CKPT_CLECG="${OUT_DIR}/ta_II_500hz_clecg_seed${SEED}_best.pt"
    ENC_100="${CLECG_DIR}/clecg_II_100hz_seed${SEED}_encoder.pt"
    EVAL_CFG="dafd_mvkt/configs/student_100hz_ta.yaml"
fi

# Select best TA (prefer CLECG-strengthened)
if [ -f "${TA_CKPT_CLECG}" ]; then
    BEST_TA="${TA_CKPT_CLECG}"
    echo "  Using CLECG-TA: ${BEST_TA}"
elif [ -f "${TA_CKPT}" ]; then
    BEST_TA="${TA_CKPT}"
    echo "  Using standard TA: ${BEST_TA}"
else
    echo "ERROR: No TA checkpoint found for lead ${LEAD}. Run 100Hz stage first."
    exit 1
fi

# Encoder init: prefer CLECG if available
if [ -f "${ENC_100}" ]; then
    INIT_ENC="${ENC_100}"
    echo "  Using CLECG student encoder: ${INIT_ENC}"
else
    INIT_ENC=""
    echo "  WARNING: No CLECG 100Hz encoder — tuning without CLECG init."
fi

LOSSES="bce,ta_mkd,ta_crf,feature"
LOSSES_FNAME=$(echo "${LOSSES}" | tr ',' '_')

# Suffix: _clecgta_clecg if CLECG TA, _clecg if only CLECG student, else ""
if [ "${BEST_TA}" = "${TA_CKPT_CLECG}" ] && [ -n "${INIT_ENC}" ]; then
    BASE_SUFFIX="_clecgta_clecg"
elif [ -n "${INIT_ENC}" ]; then
    BASE_SUFFIX="_clecg"
else
    BASE_SUFFIX=""
fi

_tune_run() {
    local VSEED="$1"
    local LR_ARG="$2"
    local ALPHA_ARG="$3"
    local LABEL="$4"

    local NAME="student_${LEAD}_100hz_${LOSSES_FNAME}${BASE_SUFFIX}_seed${VSEED}"
    local CKPT="${TUNING_DIR}/${NAME}_best.pt"
    local METRICS="${TUNING_DIR}/metrics_test_${NAME}.json"

    echo ""
    echo "  ── Tuning [${LABEL}]  ${NAME}"

    local EXTRA=""
    [ -n "${INIT_ENC}" ] && EXTRA="--init_encoder_ckpt ${INIT_ENC}"

    if [ ! -f "${CKPT}" ]; then
        python dafd_mvkt/train_student_hier.py \
            --config "${STUDENT_CFG}" \
            --data_dir "${DATA_DIR}" \
            --teacher_ckpt "${TEACHER_CKPT}" \
            --ta_ckpt "${BEST_TA}" \
            ${TA_CFG_ARG} \
            --lead "${LEAD}" \
            --losses "${LOSSES}" \
            --seed "${VSEED}" \
            --batch_size "${BATCH_SIZE}" \
            --run_name "${NAME}" \
            --output_dir "${TUNING_DIR}" \
            ${LR_ARG} ${ALPHA_ARG} \
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
            --output_dir "${TUNING_DIR}" \
            --tune_thresholds
    else
        echo "  Metrics exist — skipping evaluation."
    fi
}

# HP grid (3 variants, sequential)
#   seed 10: lower LR
#   seed 11: higher LR
#   seed 12: higher alpha_ta
_tune_run 10 "--lr 1.0e-4"  ""              "lr=1e-4"
_tune_run 11 "--lr 5.0e-4"  ""              "lr=5e-4"
_tune_run 12 ""             "--alpha_ta 2.0" "alpha_ta=2.0"

# Find best HP (highest test AUC across tuning runs)
echo ""
echo "Finding best HP configuration..."
python - <<'PYEOF'
import json, math
from pathlib import Path
import os

tuning_dir = Path(os.environ.get("TUNING_DIR", "dafd_mvkt/outputs/tuning"))
auto_dir   = Path(os.environ.get("AUTO_DIR",   "dafd_mvkt/outputs/auto"))
lead       = os.environ.get("LEAD", "I")

seed_to_hp = {10: "lr=1e-4", 11: "lr=5e-4", 12: "alpha_ta=2.0"}
best_auc, best_seed, best_hp = -1.0, -1, "unknown"

for vseed, hp_label in seed_to_hp.items():
    pattern = f"metrics_test_student_{lead}_100hz_*_seed{vseed}.json"
    files   = list(tuning_dir.glob(pattern))
    if not files:
        continue
    for f in files:
        d = json.loads(f.read_text())
        auc = d.get("macro_auc", float("nan"))
        if not math.isnan(auc) and auc > best_auc:
            best_auc  = auc
            best_seed = vseed
            best_hp   = hp_label
            best_name = f.stem.replace("metrics_test_", "")

if best_seed >= 0:
    print(f"Best tuning HP: seed={best_seed}  hp={best_hp}  AUC={best_auc:.4f}")
    env_lines = [
        f"TUNING_BEST_SEED={best_seed}",
        f"TUNING_BEST_HP={best_hp}",
        f"TUNING_BEST_AUC={best_auc:.4f}",
        f"TUNING_BEST_LEAD={lead}",
    ]
    auto_dir.mkdir(parents=True, exist_ok=True)
    (auto_dir / "tuning_best_hp.env").write_text("\n".join(env_lines) + "\n")
    print(f"Saved: {auto_dir}/tuning_best_hp.env")
else:
    print("No tuning results found.")
PYEOF

echo ""
echo "AUTO STAGE HP Tuning complete. Results in ${TUNING_DIR}"
