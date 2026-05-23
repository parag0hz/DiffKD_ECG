#!/usr/bin/env bash
# Run full-loss variants for the best skip-fork structure.
# Run AFTER screening and weight tuning complete.
#
# Usage:
#   SF_ID=SF02 WEIGHTS=0.2,0.4,0.4 DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#   bash dafd_mvkt/scripts/run_best_skip_fork_full.sh
#
# Variants:
#   B1: best skip-fork KD-only (already done from screening)
#   B2: best skip-fork + CRF
#   B3: best skip-fork + FeatureKD
#   B4: best skip-fork + CRF + FeatureKD
#
# Rule: CRF/FeatureKD applied to the closest 1L branch teacher (prefer 1L100 > 1L500).
# Not applied to skip teacher t1 in this version.

set -euo pipefail

SF_ID=${SF_ID:-"SF02"}
WEIGHTS=${WEIGHTS:-"0.20,0.40,0.40"}
DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}

CONFIG="dafd_mvkt/configs/skip_fork_join_6.yaml"
OUT_DIR="dafd_mvkt/outputs/skipfork"

echo "=============================================="
echo "  Best Skip-Fork Full-Loss Variants"
echo "  SF_ID=${SF_ID}  WEIGHTS=${WEIGHTS}  SEED=${SEED}"
echo "=============================================="

activate_env() {
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate ecg 2>/dev/null || true
    fi
}
activate_env

case "${SF_ID}" in
    SF01) T1="12L100"; T2="12L50";  T3="1L100" ;;
    SF02) T1="12L100"; T2="1L500";  T3="1L100" ;;
    SF03) T1="12L500"; T2="12L50";  T3="1L100" ;;
    SF04) T1="12L500"; T2="1L500";  T3="1L100" ;;
    SF05) T1="12L100"; T2="12L50";  T3="1L50"  ;;
    SF06) T1="1L500";  T2="1L100";  T3="1L50"  ;;
    *) echo "Unknown SF_ID=${SF_ID}"; exit 1 ;;
esac

B2_CKPT="${OUT_DIR}/${SF_ID}_branch2_${T2}_from_${T1}_seed${SEED}_best.pt"
B3_CKPT="${OUT_DIR}/${SF_ID}_branch3_${T3}_from_${T1}_seed${SEED}_best.pt"
SIMCLR_CKPT="dafd_mvkt/outputs/simclr/simclr_II_50hz_seed${SEED}_encoder.pt"

case "${T1}" in
    12L500) T1_CKPT="dafd_mvkt/outputs/teacher_best.pt" ;;
    12L100) T1_CKPT="dafd_mvkt/outputs/teacher_12lead_100hz_resnet34_best.pt" ;;
    12L50)  T1_CKPT="dafd_mvkt/outputs/teacher_12lead_50hz_resnet34_best.pt" ;;
    1L500)  T1_CKPT="dafd_mvkt/outputs/ta_II_500hz_simclr_seed${SEED}_best.pt" ;;
    1L100)  T1_CKPT="dafd_mvkt/outputs/ta_II_100hz_from_ta500_seed${SEED}_best.pt" ;;
    *) echo "Unknown T1=${T1}"; exit 1 ;;
esac

for f in "${B2_CKPT}" "${B3_CKPT}"; do
    if [ ! -f "${f}" ]; then
        echo "[ERROR] Checkpoint not found: ${f}"
        echo "Run run_skip_fork_6.sh first."
        exit 1
    fi
done

# Select preferred branch for CRF/FeatureKD (prefer 1L100 > 1L500 > 1L50)
CRF_VIEW="${T3}"
CRF_CKPT="${B3_CKPT}"
if [[ "${T2}" == "1L100" ]]; then
    CRF_VIEW="${T2}"; CRF_CKPT="${B2_CKPT}"
elif [[ "${T2}" == "1L500" && "${T3}" != "1L100" ]]; then
    CRF_VIEW="${T2}"; CRF_CKPT="${B2_CKPT}"
fi

echo "Branch for CRF/FeatureKD: ${CRF_VIEW}  ckpt: ${CRF_CKPT}"

INIT_ARG=""
if [ -f "${SIMCLR_CKPT}" ]; then
    INIT_ARG="--student_init_ckpt ${SIMCLR_CKPT}"
fi

run_variant() {
    local VARIANT=$1
    local LOSSES=$2
    local OUT_NAME="student_II_50hz_skipfork_${SF_ID}_${T1}_to_${T2}_${T3}_${VARIANT}_seed${SEED}"
    local METRICS="${OUT_DIR}/metrics_test_${OUT_NAME}.json"

    echo ""
    echo "── ${VARIANT}: losses=${LOSSES}"

    if [ -f "${METRICS}" ]; then
        echo "   [skip] already exists"
        return
    fi

    python dafd_mvkt/train_student.py \
        --config         dafd_mvkt/configs/student_50hz.yaml \
        --data_dir       "${DATA_DIR}" \
        --teacher2_ckpt  "${CRF_CKPT}" \
        --teacher3_ckpt  "${B3_CKPT}" \
        --skip_ckpt      "${T1_CKPT}" \
        --teacher2_view  "${CRF_VIEW}" \
        --teacher3_view  "${T3}" \
        --skip_view      "${T1}" \
        --teacher_weights "${WEIGHTS}" \
        --losses         "${LOSSES}" \
        --output_dir     "${OUT_DIR}" \
        --output_name    "${OUT_NAME}" \
        ${INIT_ARG} \
        --seed           "${SEED}"

    python dafd_mvkt/evaluate.py \
        --config     dafd_mvkt/configs/student_50hz.yaml \
        --data_dir   "${DATA_DIR}" \
        --ckpt       "${OUT_DIR}/${OUT_NAME}_best.pt" \
        --split      test \
        --model_name "${OUT_NAME}" \
        --output_dir "${OUT_DIR}" \
        --tune_thresholds
}

run_variant "kd"             "bce,mkd_skip_fork"
run_variant "kd_crf"         "bce,mkd_skip_fork,crf"
run_variant "kd_feature"     "bce,mkd_skip_fork,feature"
run_variant "kd_crf_feature" "bce,mkd_skip_fork,crf,feature"

# aggregate
echo ""
echo "Aggregating full-loss results …"
python - << 'PYEOF'
import json, csv
from pathlib import Path
import os

SF_ID   = os.environ.get("SF_ID", "SF02")
SEED    = int(os.environ.get("SEED", 0))
OUT_DIR = Path("dafd_mvkt/outputs/skipfork")

C14_AUC = 0.8425; MVKT_AUC = 0.843

variants = ["kd", "kd_crf", "kd_feature", "kd_crf_feature"]
rows = []
for v in variants:
    pat = f"metrics_test_student_II_50hz_skipfork_{SF_ID}_*_{v}_seed{SEED}.json"
    mfs = list(OUT_DIR.glob(pat))
    if not mfs:
        rows.append({"variant": v, "auc": "N/A", "f1_tuned": "N/A", "status": "missing"})
        continue
    d   = json.load(open(mfs[0]))
    auc = d["macro_auc"]
    f1t = d.get("macro_f1_tuned", d.get("macro_f1", 0))
    rows.append({
        "variant": v, "sf_id": SF_ID,
        "auc": round(auc, 4), "f1_tuned": round(f1t, 4),
        "delta_auc_vs_c14": round(auc - C14_AUC, 4),
        "beats_mvkt": "YES" if auc > MVKT_AUC else "no",
        "status": "done",
    })

csv_p = OUT_DIR / "best_skip_full_results.csv"
with open(csv_p, "w", newline="") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

md_p = OUT_DIR / "best_skip_full_results.md"
with open(md_p, "w") as f:
    f.write(f"# Best Skip-Fork ({SF_ID}) Full-Loss Variants\n\n")
    f.write("| Variant | AUC | F1_tuned | ΔAUC vs C14 | >MVKT |\n")
    f.write("|---------|-----|----------|-------------|-------|\n")
    for r in rows:
        if r["status"] == "done":
            f.write(f"| {r['variant']} | {r['auc']} | {r['f1_tuned']} | "
                    f"{r['delta_auc_vs_c14']:+.4f} | {r['beats_mvkt']} |\n")
        else:
            f.write(f"| {r['variant']} | N/A | N/A | - | - |\n")

print(f"Saved: {csv_p}")
print(f"Saved: {md_p}")
PYEOF

echo ""
echo "Done. cat dafd_mvkt/outputs/skipfork/best_skip_full_results.md"
