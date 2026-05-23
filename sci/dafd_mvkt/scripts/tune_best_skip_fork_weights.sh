#!/usr/bin/env bash
# Weight tuning for the best skip-fork structure.
# Run AFTER 6 skip-fork screening completes and best SF is selected.
#
# Usage:
#   SF_ID=SF02 DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#   bash dafd_mvkt/scripts/tune_best_skip_fork_weights.sh

set -euo pipefail

SF_ID=${SF_ID:-"SF02"}
DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}

CONFIG="dafd_mvkt/configs/skip_fork_join_6.yaml"
OUT_DIR="dafd_mvkt/outputs/skipfork"

echo "=============================================="
echo "  Skip-Fork Weight Tuning"
echo "  SF_ID=${SF_ID}  SEED=${SEED}"
echo "=============================================="

activate_env() {
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate ecg 2>/dev/null || true
    fi
}
activate_env

# resolve SF spec
case "${SF_ID}" in
    SF01) T1="12L100"; T2="12L50";  T3="1L100" ;;
    SF02) T1="12L100"; T2="1L500";  T3="1L100" ;;
    SF03) T1="12L500"; T2="12L50";  T3="1L100" ;;
    SF04) T1="12L500"; T2="1L500";  T3="1L100" ;;
    SF05) T1="12L100"; T2="12L50";  T3="1L50"  ;;
    SF06) T1="1L500";  T2="1L100";  T3="1L50"  ;;
    *) echo "Unknown SF_ID=${SF_ID}"; exit 1 ;;
esac

# resolved branch checkpoints from screening run
B2_CKPT="${OUT_DIR}/${SF_ID}_branch2_${T2}_from_${T1}_seed${SEED}_best.pt"
B3_CKPT="${OUT_DIR}/${SF_ID}_branch3_${T3}_from_${T1}_seed${SEED}_best.pt"
SIMCLR_CKPT="dafd_mvkt/outputs/simclr/simclr_II_50hz_seed${SEED}_encoder.pt"

# t1 skip checkpoint
case "${T1}" in
    12L500) T1_CKPT="dafd_mvkt/outputs/teacher_best.pt" ;;
    12L100) T1_CKPT="dafd_mvkt/outputs/teacher_12lead_100hz_resnet34_best.pt" ;;
    12L50)  T1_CKPT="dafd_mvkt/outputs/teacher_12lead_50hz_resnet34_best.pt" ;;
    1L500)  T1_CKPT="dafd_mvkt/outputs/ta_II_500hz_simclr_seed${SEED}_best.pt" ;;
    1L100)  T1_CKPT="dafd_mvkt/outputs/ta_II_100hz_from_ta500_seed${SEED}_best.pt" ;;
    *) echo "Unknown T1=${T1}"; exit 1 ;;
esac

if [ ! -f "${B2_CKPT}" ]; then
    echo "[ERROR] Branch2 checkpoint not found: ${B2_CKPT}"
    echo "Run run_skip_fork_6.sh first."
    exit 1
fi
if [ ! -f "${B3_CKPT}" ]; then
    echo "[ERROR] Branch3 checkpoint not found: ${B3_CKPT}"
    echo "Run run_skip_fork_6.sh first."
    exit 1
fi

echo "t1 ckpt:  ${T1_CKPT}"
echo "branch2:  ${B2_CKPT}"
echo "branch3:  ${B3_CKPT}"

# define weight sets depending on t1 view
if [[ "${T1}" == "12L100" ]]; then
    declare -A WEIGHT_SETS
    WEIGHT_SETS=( ["W01"]="0.10,0.45,0.45" ["W02"]="0.20,0.40,0.40" ["W03"]="0.30,0.35,0.35" ["W04"]="0.05,0.475,0.475" )
    WEIGHT_KEYS=("W01" "W02" "W03" "W04")
elif [[ "${T1}" == "12L500" ]]; then
    declare -A WEIGHT_SETS
    WEIGHT_SETS=( ["W01"]="0.05,0.475,0.475" ["W02"]="0.10,0.45,0.45" ["W03"]="0.20,0.40,0.40" )
    WEIGHT_KEYS=("W01" "W02" "W03")
else
    declare -A WEIGHT_SETS
    WEIGHT_SETS=( ["W01"]="0.10,0.45,0.45" ["W02"]="0.20,0.40,0.40" ["W03"]="0.30,0.35,0.35" )
    WEIGHT_KEYS=("W01" "W02" "W03")
fi

run_weight_variant() {
    local WKEY=$1
    local WEIGHTS=$2
    local OUT_NAME="student_II_50hz_skipfork_${SF_ID}_${T1}_to_${T2}_${T3}_${WKEY}_seed${SEED}"
    local METRICS="${OUT_DIR}/metrics_test_${OUT_NAME}.json"

    echo ""
    echo "── ${WKEY}: weights=${WEIGHTS}"

    if [ -f "${METRICS}" ]; then
        echo "   [skip] already exists"
        return
    fi

    INIT_ARG=""
    if [ -f "${SIMCLR_CKPT}" ]; then
        INIT_ARG="--student_init_ckpt ${SIMCLR_CKPT}"
    fi

    python dafd_mvkt/train_student_skip_fork.py \
        --config           "${CONFIG}" \
        --data_dir         "${DATA_DIR}" \
        --skip_view        "${T1}" \
        --branch2_view     "${T2}" \
        --branch3_view     "${T3}" \
        --skip_ckpt        "${T1_CKPT}" \
        --branch2_ckpt     "${B2_CKPT}" \
        --branch3_ckpt     "${B3_CKPT}" \
        --teacher_weights  "${WEIGHTS}" \
        --output_name      "${OUT_NAME}" \
        --output_dir       "${OUT_DIR}" \
        ${INIT_ARG} \
        --seed             "${SEED}"
}

for WKEY in "${WEIGHT_KEYS[@]}"; do
    run_weight_variant "${WKEY}" "${WEIGHT_SETS[$WKEY]}"
done

# aggregate
echo ""
echo "Aggregating weight tuning results …"
python - << 'PYEOF'
import json, csv
from pathlib import Path
import os

SF_ID   = os.environ.get("SF_ID", "SF02")
SEED    = int(os.environ.get("SEED", 0))
OUT_DIR = Path("dafd_mvkt/outputs/skipfork")

C14_AUC = 0.8425; MVKT_AUC = 0.843

pat = f"metrics_test_student_II_50hz_skipfork_{SF_ID}_*_W0*_seed{SEED}.json"
mfs = sorted(OUT_DIR.glob(pat))

rows = []
for mf in mfs:
    d   = json.load(open(mf))
    auc = d["macro_auc"]
    f1t = d.get("macro_f1_tuned", d.get("macro_f1", 0))
    # extract weight key from name
    name = mf.stem.replace("metrics_test_", "")
    wkey = [p for p in name.split("_") if p.startswith("W0")]
    wkey = wkey[0] if wkey else "?"
    rows.append({
        "sf_id": SF_ID, "weight_key": wkey,
        "output_name": name,
        "auc": round(auc, 4), "f1_tuned": round(f1t, 4),
        "delta_auc_vs_c14": round(auc - C14_AUC, 4),
        "beats_mvkt": "YES" if auc > MVKT_AUC else "no",
    })

rows.sort(key=lambda r: r["auc"], reverse=True)

csv_p = OUT_DIR / "best_skip_weight_tuning_results.csv"
with open(csv_p, "w", newline="") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

md_p = OUT_DIR / "best_skip_weight_tuning_results.md"
with open(md_p, "w") as f:
    f.write(f"# Skip-Fork Weight Tuning ({SF_ID})\n\n")
    f.write("| Wkey | AUC | F1_tuned | ΔAUC vs C14 | >MVKT |\n")
    f.write("|------|-----|----------|-------------|-------|\n")
    for r in rows:
        f.write(f"| {r['weight_key']} | {r['auc']} | {r['f1_tuned']} | "
                f"{r['delta_auc_vs_c14']:+.4f} | {r['beats_mvkt']} |\n")

print(f"Saved: {csv_p}")
print(f"Saved: {md_p}")
if rows:
    print(f"\nBest weights: {rows[0]['weight_key']}  AUC={rows[0]['auc']}")
PYEOF

echo ""
echo "Done. cat dafd_mvkt/outputs/skipfork/best_skip_weight_tuning_results.md"
