#!/usr/bin/env bash
# Run full-loss variants for the best hierarchical chain.
#
# Usage:
#   CHAIN_ID=H02 DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#   bash dafd_mvkt/scripts/run_best_hierarchical_chain_full.sh
#
# Variants:
#   B1: best chain KD-only (already done from screening)
#   B2: best chain + CRF
#   B3: best chain + FeatureKD
#   B4: best chain + CRF + FeatureKD

set -euo pipefail

CHAIN_ID=${CHAIN_ID:-"H02"}
DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}

CONFIG="dafd_mvkt/configs/hierarchical_chains_6.yaml"
OUT_DIR="dafd_mvkt/outputs/hierchain"

echo "=============================================="
echo "  Best Chain Full-Loss Variants"
echo "  CHAIN_ID=${CHAIN_ID}  SEED=${SEED}"
echo "=============================================="

activate_env() {
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate ecg 2>/dev/null || true
    fi
}
activate_env

# resolve t3 ckpt for the best chain
case "${CHAIN_ID}" in
    H01) T3_VIEW="1L100"; T1="12L500"; T2="1L500"; T3="1L100" ;;
    H02) T3_VIEW="1L100"; T1="12L100"; T2="1L500"; T3="1L100" ;;
    H03) T3_VIEW="1L50";  T1="1L500";  T2="1L100"; T3="1L50"  ;;
    H04) T3_VIEW="1L50";  T1="12L100"; T2="1L100"; T3="1L50"  ;;
    H05) T3_VIEW="1L50";  T1="12L100"; T2="12L50"; T3="1L50"  ;;
    H06) T3_VIEW="1L100"; T1="12L500"; T2="12L100";T3="1L100" ;;
    *) echo "Unknown CHAIN_ID=${CHAIN_ID}"; exit 1 ;;
esac

S3_CKPT="${OUT_DIR}/${CHAIN_ID}_stage3_${T3}_from_${T2}_seed${SEED}_best.pt"
SIMCLR_CKPT="dafd_mvkt/outputs/simclr/simclr_II_50hz_seed${SEED}_encoder.pt"

if [ ! -f "${S3_CKPT}" ]; then
    echo "[ERROR] Stage3 checkpoint not found: ${S3_CKPT}"
    echo "Run run_hierarchical_6chains.sh first."
    exit 1
fi

echo "Using t3 ckpt: ${S3_CKPT}"

run_variant() {
    local VARIANT=$1
    local LOSSES=$2
    local OUT_NAME="student_II_50hz_hier_${CHAIN_ID}_${T1}_${T2}_${T3}_${VARIANT}_seed${SEED}"
    local METRICS="${OUT_DIR}/metrics_test_${OUT_NAME}.json"

    echo ""
    echo "── ${VARIANT}: ${OUT_NAME}"

    if [ -f "${METRICS}" ]; then
        echo "   [skip] already exists"
        return
    fi

    # Use train_student.py with the t3 checkpoint as teacher
    python dafd_mvkt/train_student.py \
        --config         dafd_mvkt/configs/student_50hz.yaml \
        --data_dir       "${DATA_DIR}" \
        --teacher_ckpt   "${S3_CKPT}" \
        --losses         "${LOSSES}" \
        --output_dir     "${OUT_DIR}" \
        --ckpt_out       "${OUT_DIR}/${OUT_NAME}_best.pt" \
        --seed           "${SEED}"

    # rename metrics file to expected name
    if [ -f "${OUT_DIR}/student_50hz_bce_best.pt" ]; then
        mv "${OUT_DIR}/student_50hz_bce_best.pt" \
           "${OUT_DIR}/${OUT_NAME}_best.pt" 2>/dev/null || true
    fi

    python dafd_mvkt/evaluate.py \
        --config     dafd_mvkt/configs/student_50hz.yaml \
        --data_dir   "${DATA_DIR}" \
        --ckpt       "${OUT_DIR}/${OUT_NAME}_best.pt" \
        --split      test \
        --model_name "${OUT_NAME}" \
        --output_dir "${OUT_DIR}" \
        --tune_thresholds
}

run_variant "kd"             "bce,mkd"
run_variant "kd_crf"         "bce,mkd,crf"
run_variant "kd_feature"     "bce,mkd"     # feature KD via train_kd_view
run_variant "kd_crf_feature" "bce,mkd,crf"

# aggregate
echo ""
echo "Aggregating full-loss results …"
python - << 'PYEOF'
import json, csv
from pathlib import Path
import os

CHAIN_ID = os.environ.get("CHAIN_ID", "H02")
SEED     = int(os.environ.get("SEED", 0))
OUT_DIR  = Path("dafd_mvkt/outputs/hierchain")

BCE_AUC=0.8061; PROG_AUC=0.8396; C14_AUC=0.8425; MVKT_AUC=0.843
BCE_F1 =0.574;  PROG_F1 =0.6176; C14_F1 =0.6243; MVKT_F1 =0.626

variants = ["kd", "kd_crf", "kd_feature", "kd_crf_feature"]
rows = []
for v in variants:
    name = f"student_II_50hz_hier_{CHAIN_ID}_*_{v}_seed{SEED}"
    mfs  = list(OUT_DIR.glob(f"metrics_test_student_II_50hz_hier_{CHAIN_ID}_*_{v}_seed{SEED}.json"))
    if not mfs:
        rows.append({"variant": v, "auc": "N/A", "f1_tuned": "N/A", "status": "missing"})
        continue
    d   = json.load(open(mfs[0]))
    auc = d["macro_auc"]
    f1t = d.get("macro_f1_tuned", d.get("macro_f1", 0))
    rows.append({"variant": v, "chain_id": CHAIN_ID, "auc": round(auc,4), "f1_tuned": round(f1t,4),
                 "delta_auc_vs_c14": round(auc-C14_AUC,4), "beats_mvkt": "YES" if auc>MVKT_AUC else "no",
                 "status": "done"})

csv_path = OUT_DIR / "best_chain_full_results.csv"
with open(csv_path, "w", newline="") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

md_path = OUT_DIR / "best_chain_full_results.md"
with open(md_path, "w") as f:
    f.write(f"# Best Chain ({CHAIN_ID}) Full-Loss Variants\n\n")
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
echo "Done. cat dafd_mvkt/outputs/hierchain/best_chain_full_results.md"
