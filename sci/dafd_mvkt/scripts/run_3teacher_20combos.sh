#!/usr/bin/env bash
# Run all 20 teacher-combination screenings for 1-lead 50Hz student KD.
#
# Usage:
#   DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#   bash dafd_mvkt/scripts/run_3teacher_20combos.sh

set -euo pipefail
DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}
OUT_DIR="dafd_mvkt/outputs"
MULTI_DIR="${OUT_DIR}/multiteacher"
mkdir -p "${MULTI_DIR}"

BANK_JSON="dafd_mvkt/configs/teacher_bank_II.json"
CONFIG="dafd_mvkt/configs/student_50hz_3teacher.yaml"
SIMCLR_CKPT="${OUT_DIR}/simclr/simclr_II_50hz_seed0_encoder.pt"

echo "=============================================="
echo "  3-Teacher 20-Combo Screening"
echo "  DATA_DIR=${DATA_DIR}  LEAD=${LEAD}  SEED=${SEED}"
echo "=============================================="

activate_env() {
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate ecg 2>/dev/null || true
    fi
}
activate_env

# ── Step 1: teacher bank ───────────────────────────────────────────────────
echo ""
echo "Step 1: Preparing teacher bank …"
DATA_DIR="${DATA_DIR}" SEED="${SEED}" LEAD="${LEAD}" \
    bash dafd_mvkt/scripts/prepare_3teacher_bank.sh

# ── Step 2: check SimCLR init ──────────────────────────────────────────────
echo ""
echo "Step 2: Checking SimCLR student init …"
if [ ! -f "${SIMCLR_CKPT}" ]; then
    echo "  [WARN] SimCLR encoder not found: ${SIMCLR_CKPT}"
    echo "  Running SimCLR pretraining for 1-lead II 50Hz …"
    python dafd_mvkt/pretrain_simclr.py \
        --data_dir "${DATA_DIR}" \
        --lead "${LEAD}" \
        --sampling_rate 50 \
        --seed "${SEED}" \
        --output_dir "${OUT_DIR}/simclr"
fi
echo "  SimCLR init: ${SIMCLR_CKPT}"

# ── Step 3: run 20 combos ──────────────────────────────────────────────────
COMBOS=(C01 C02 C03 C04 C05 C06 C07 C08 C09 C10
        C11 C12 C13 C14 C15 C16 C17 C18 C19 C20)

echo ""
echo "Step 3: Running 20 combos …"
for COMBO in "${COMBOS[@]}"; do
    OUT_NAME="student_II_50hz_3T_${COMBO}_equal_simclr_seed${SEED}"
    METRICS="${OUT_DIR}/metrics_test_${OUT_NAME}.json"
    CKPT="${OUT_DIR}/${OUT_NAME}_best.pt"

    echo ""
    echo "── ${COMBO} → ${OUT_NAME}"

    if [ -f "${METRICS}" ]; then
        echo "   [skip] metrics already exist"
        continue
    fi

    if [ -f "${CKPT}" ] && [ ! -f "${METRICS}" ]; then
        echo "   [eval-only] checkpoint exists, running evaluation …"
        python dafd_mvkt/evaluate.py \
            --config   "${CONFIG}" \
            --data_dir "${DATA_DIR}" \
            --ckpt     "${CKPT}" \
            --split    test \
            --model_name "${OUT_NAME}" \
            --output_dir "${OUT_DIR}" \
            --tune_thresholds
        continue
    fi

    python dafd_mvkt/train_student_3teacher.py \
        --config             "${CONFIG}" \
        --data_dir           "${DATA_DIR}" \
        --lead               "${LEAD}" \
        --teacher_combo      "${COMBO}" \
        --teacher_ckpts_json "${BANK_JSON}" \
        --lambda_kd          1.0 \
        --temperature        2.0 \
        --student_init_ckpt  "${SIMCLR_CKPT}" \
        --seed               "${SEED}" \
        --output_name        "${OUT_NAME}"

done

# ── Step 4: aggregate results ──────────────────────────────────────────────
echo ""
echo "Step 4: Aggregating results …"
python - << 'PYEOF'
import json, csv, os
from pathlib import Path

COMBOS = [f"C{i:02d}" for i in range(1, 21)]
COMBO_TEACHERS = {
    "C01":["T1","T2","T3"], "C02":["T1","T2","T4"], "C03":["T1","T2","T5"],
    "C04":["T1","T2","T6"], "C05":["T1","T3","T4"], "C06":["T1","T3","T5"],
    "C07":["T1","T3","T6"], "C08":["T1","T4","T5"], "C09":["T1","T4","T6"],
    "C10":["T1","T5","T6"], "C11":["T2","T3","T4"], "C12":["T2","T3","T5"],
    "C13":["T2","T3","T6"], "C14":["T2","T4","T5"], "C15":["T2","T4","T6"],
    "C16":["T2","T5","T6"], "C17":["T3","T4","T5"], "C18":["T3","T4","T6"],
    "C19":["T3","T5","T6"], "C20":["T4","T5","T6"],
}
TEACHER_LABELS = {
    "T1":"12L500","T2":"12L100","T3":"12L50",
    "T4":"1L500","T5":"1L100","T6":"1L50",
}

BCE_AUC     = 0.8061;  BCE_F1     = 0.5740
BEST_AUC    = 0.8396;  BEST_F1    = 0.6176
MVKT_AUC    = 0.843;   MVKT_F1    = 0.626

SEED = int(os.environ.get("SEED", 0))
OUT_DIR = Path("dafd_mvkt/outputs")
MULTI_DIR = OUT_DIR / "multiteacher"

rows = []
for combo in COMBOS:
    out_name = f"student_II_50hz_3T_{combo}_equal_simclr_seed{SEED}"
    mf = OUT_DIR / f"metrics_test_{out_name}.json"
    teachers = COMBO_TEACHERS[combo]
    tlabels  = "+".join(TEACHER_LABELS[t] for t in teachers)

    if not mf.exists():
        rows.append({"combo_id": combo, "teachers": tlabels,
                     "auc": "N/A", "f1_tuned": "N/A", "f1_0_5": "N/A",
                     "delta_auc_vs_bce": "N/A", "delta_f1_vs_bce": "N/A",
                     "delta_auc_vs_best": "N/A", "delta_f1_vs_best": "N/A",
                     "beats_current_50hz_best": "N/A",
                     "beats_mvkt_auc": "N/A", "beats_mvkt_f1": "N/A",
                     "status": "missing"})
        continue

    d   = json.load(open(mf))
    auc = d["macro_auc"]
    f1t = d.get("macro_f1_tuned", d.get("macro_f1", 0))
    f10 = d.get("macro_f1_0_5", 0)

    rows.append({
        "combo_id": combo,
        "teachers": tlabels,
        "auc":       round(auc, 4),
        "f1_tuned":  round(f1t, 4),
        "f1_0_5":    round(f10, 4),
        "delta_auc_vs_bce":   round(auc - BCE_AUC,  4),
        "delta_f1_vs_bce":    round(f1t - BCE_F1,   4),
        "delta_auc_vs_best":  round(auc - BEST_AUC, 4),
        "delta_f1_vs_best":   round(f1t - BEST_F1,  4),
        "beats_current_50hz_best": "YES" if auc > BEST_AUC else "no",
        "beats_mvkt_auc": "YES" if auc > MVKT_AUC else "no",
        "beats_mvkt_f1":  "YES" if f1t > MVKT_F1  else "no",
        "status": "done",
    })

# sort by AUC descending (N/A last)
done_rows    = [r for r in rows if r["status"] == "done"]
missing_rows = [r for r in rows if r["status"] == "missing"]
done_rows.sort(key=lambda r: r["auc"], reverse=True)
rows_sorted  = done_rows + missing_rows

csv_path = MULTI_DIR / "3teacher_20combos_results.csv"
with open(csv_path, "w", newline="") as f:
    if rows_sorted:
        w = csv.DictWriter(f, fieldnames=rows_sorted[0].keys())
        w.writeheader(); w.writerows(rows_sorted)

md_path = MULTI_DIR / "3teacher_20combos_results.md"
with open(md_path, "w") as f:
    f.write("# 3-Teacher 20-Combo Screening Results\n\n")
    f.write(f"References:\n")
    f.write(f"- BCE 50Hz:      AUC={BCE_AUC}  F1={BCE_F1}\n")
    f.write(f"- Current best:  AUC={BEST_AUC} F1={BEST_F1} (Progressive+SimCLR)\n")
    f.write(f"- MVKT target:   AUC={MVKT_AUC} F1={MVKT_F1}\n\n")
    f.write("| Rank | Combo | Teachers | AUC | F1_tuned | ΔAUC vs BCE | ΔAUC vs Best | >MVKT |\n")
    f.write("|------|-------|----------|-----|----------|-------------|--------------|-------|\n")
    for rank, r in enumerate(done_rows, 1):
        mvkt = "✅" if r["beats_mvkt_auc"] == "YES" else ""
        f.write(f"| {rank} | {r['combo_id']} | {r['teachers']} | {r['auc']} | "
                f"{r['f1_tuned']} | {r['delta_auc_vs_bce']:+.4f} | "
                f"{r['delta_auc_vs_best']:+.4f} | {mvkt} |\n")
    if missing_rows:
        f.write(f"\n**Missing ({len(missing_rows)}):** " +
                ", ".join(r["combo_id"] for r in missing_rows) + "\n")

print(f"Saved: {csv_path}")
print(f"Saved: {md_path}")
print(f"\nResults ({len(done_rows)}/{len(COMBOS)} done):")
for r in done_rows[:5]:
    print(f"  {r['combo_id']} {r['teachers']}: AUC={r['auc']} F1={r['f1_tuned']}")
PYEOF

echo ""
echo "=============================================="
echo "  All 20 combos complete."
echo "  cat dafd_mvkt/outputs/multiteacher/3teacher_20combos_results.md"
echo "=============================================="
