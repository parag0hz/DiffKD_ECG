#!/usr/bin/env bash
# Run full-loss variants for the best fork-join structure.
#
# Usage:
#   FORK_ID=F02 DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#   bash dafd_mvkt/scripts/run_best_fork_join_full.sh
#
# Variants:
#   B1: best fork KD-only (already done from screening)
#   B2: best fork + CRF
#   B3: best fork + FeatureKD
#   B4: best fork + CRF + FeatureKD

set -euo pipefail

FORK_ID=${FORK_ID:-"F02"}
DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}

CONFIG="dafd_mvkt/configs/fork_join_dual_6.yaml"
OUT_DIR="dafd_mvkt/outputs/forkjoin"

echo "=============================================="
echo "  Best Fork-Join Full-Loss Variants"
echo "  FORK_ID=${FORK_ID}  SEED=${SEED}"
echo "=============================================="

activate_env() {
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate ecg 2>/dev/null || true
    fi
}
activate_env

# resolve fork spec
case "${FORK_ID}" in
    F01) T1="12L100"; T2="12L50";  T3="1L100" ;;
    F02) T1="12L100"; T2="1L500";  T3="1L100" ;;
    F03) T1="12L500"; T2="12L50";  T3="1L100" ;;
    F04) T1="12L500"; T2="1L500";  T3="1L100" ;;
    F05) T1="12L100"; T2="12L50";  T3="1L50"  ;;
    F06) T1="1L500";  T2="1L100";  T3="1L50"  ;;
    *) echo "Unknown FORK_ID=${FORK_ID}"; exit 1 ;;
esac

T2_CKPT="${OUT_DIR}/${FORK_ID}_branch2_${T2}_from_${T1}_seed${SEED}_best.pt"
T3_CKPT="${OUT_DIR}/${FORK_ID}_branch3_${T3}_from_${T1}_seed${SEED}_best.pt"
SIMCLR_CKPT="dafd_mvkt/outputs/simclr/simclr_II_50hz_seed${SEED}_encoder.pt"

if [ ! -f "${T2_CKPT}" ]; then
    echo "[ERROR] Branch2 checkpoint not found: ${T2_CKPT}"
    echo "Run run_fork_join_6.sh first."
    exit 1
fi
if [ ! -f "${T3_CKPT}" ]; then
    echo "[ERROR] Branch3 checkpoint not found: ${T3_CKPT}"
    echo "Run run_fork_join_6.sh first."
    exit 1
fi

echo "Using branch2: ${T2_CKPT}"
echo "Using branch3: ${T3_CKPT}"

run_variant() {
    local VARIANT=$1
    local LOSSES=$2
    local OUT_NAME="student_II_50hz_forkjoin_${FORK_ID}_${T1}_to_${T2}_${T3}_${VARIANT}_seed${SEED}"
    local METRICS="${OUT_DIR}/metrics_test_${OUT_NAME}.json"

    echo ""
    echo "── ${VARIANT}: ${OUT_NAME}"

    if [ -f "${METRICS}" ]; then
        echo "   [skip] already exists"
        return
    fi

    python dafd_mvkt/train_student.py \
        --config         dafd_mvkt/configs/student_50hz.yaml \
        --data_dir       "${DATA_DIR}" \
        --teacher2_ckpt  "${T2_CKPT}" \
        --teacher3_ckpt  "${T3_CKPT}" \
        --teacher2_view  "${T2}" \
        --teacher3_view  "${T3}" \
        --losses         "${LOSSES}" \
        --output_dir     "${OUT_DIR}" \
        --output_name    "${OUT_NAME}" \
        --student_init   "${SIMCLR_CKPT}" \
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

run_variant "kd"             "bce,mkd_dual"
run_variant "kd_crf"         "bce,mkd_dual,crf"
run_variant "kd_feature"     "bce,mkd_dual,feature"
run_variant "kd_crf_feature" "bce,mkd_dual,crf,feature"

# aggregate
echo ""
echo "Aggregating full-loss results …"
python - << 'PYEOF'
import json, csv
from pathlib import Path
import os

FORK_ID = os.environ.get("FORK_ID", "F02")
SEED    = int(os.environ.get("SEED", 0))
OUT_DIR = Path("dafd_mvkt/outputs/forkjoin")

C14_AUC=0.8425; MVKT_AUC=0.843

variants = ["kd", "kd_crf", "kd_feature", "kd_crf_feature"]
rows = []
for v in variants:
    pat = f"metrics_test_student_II_50hz_forkjoin_{FORK_ID}_*_{v}_seed{SEED}.json"
    mfs = list(OUT_DIR.glob(pat))
    if not mfs:
        rows.append({"variant": v, "auc": "N/A", "f1_tuned": "N/A", "status": "missing"})
        continue
    d   = json.load(open(mfs[0]))
    auc = d["macro_auc"]
    f1t = d.get("macro_f1_tuned", d.get("macro_f1", 0))
    rows.append({"variant": v, "fork_id": FORK_ID, "auc": round(auc,4),
                 "f1_tuned": round(f1t,4),
                 "delta_auc_vs_c14": round(auc-C14_AUC,4),
                 "beats_mvkt": "YES" if auc>MVKT_AUC else "no",
                 "status": "done"})

csv_path = OUT_DIR / "best_fork_full_results.csv"
with open(csv_path, "w", newline="") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

md_path = OUT_DIR / "best_fork_full_results.md"
with open(md_path, "w") as f:
    f.write(f"# Best Fork ({FORK_ID}) Full-Loss Variants\n\n")
    f.write("| Variant | AUC | F1_tuned | ΔAUC vs C14 | >MVKT |\n")
    f.write("|---------|-----|----------|-------------|-------|\n")
    for r in rows:
        if r["status"] == "done":
            f.write(f"| {r['variant']} | {r['auc']} | {r['f1_tuned']} | {r['delta_auc_vs_c14']:+.4f} | {r['beats_mvkt']} |\n")
        else:
            f.write(f"| {r['variant']} | N/A | N/A | - | - |\n")

print(f"Saved: {csv_path}")
print(f"Saved: {md_path}")
PYEOF

echo ""
echo "Done. cat dafd_mvkt/outputs/forkjoin/best_fork_full_results.md"
