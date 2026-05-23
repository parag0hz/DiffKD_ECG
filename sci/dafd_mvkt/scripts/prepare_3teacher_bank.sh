#!/usr/bin/env bash
# Prepare and evaluate the 6-teacher bank for multi-teacher KD.
#
# Usage:
#   DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#   bash dafd_mvkt/scripts/prepare_3teacher_bank.sh

set -euo pipefail
DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}
OUT_DIR="dafd_mvkt/outputs"
MULTI_DIR="dafd_mvkt/outputs/multiteacher"
mkdir -p "${MULTI_DIR}"

echo "=============================================="
echo "  Teacher Bank Preparation"
echo "  DATA_DIR=${DATA_DIR}  LEAD=${LEAD}"
echo "=============================================="

activate_env() {
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate ecg
    fi
}
activate_env

# ── helper: evaluate a teacher ────────────────────────────────────────────────
eval_teacher() {
    local TID="$1"
    local CONFIG="$2"
    local CKPT="$3"
    local MODEL_NAME="$4"
    local HZ="${5:-500}"
    local EXTRA="${6:-}"

    local METRICS="${OUT_DIR}/metrics_test_${MODEL_NAME}.json"
    if [ -f "${METRICS}" ]; then
        echo "  [skip] ${TID} metrics already exist: ${METRICS}"
        return 0
    fi
    echo "  Evaluating ${TID} (${MODEL_NAME}) …"
    python dafd_mvkt/evaluate.py \
        --config   "${CONFIG}" \
        --data_dir "${DATA_DIR}" \
        --ckpt     "${CKPT}" \
        --split    test \
        --model_name "${MODEL_NAME}" \
        --output_dir "${OUT_DIR}" \
        --hz "${HZ}" \
        ${EXTRA}
}

echo ""
echo "── T1: 12-lead 500Hz ──────────────────────────"
eval_teacher T1 \
    dafd_mvkt/configs/teacher_500hz.yaml \
    "${OUT_DIR}/teacher_best.pt" \
    teacher \
    500

echo ""
echo "── T2: 12-lead 100Hz ──────────────────────────"
eval_teacher T2 \
    dafd_mvkt/configs/teacher_100hz_resnet34.yaml \
    "${OUT_DIR}/teacher_12lead_100hz_resnet34_best.pt" \
    teacher_12lead_100hz_resnet34 \
    100

echo ""
echo "── T3: 12-lead 50Hz ───────────────────────────"
eval_teacher T3 \
    dafd_mvkt/configs/teacher_50hz_resnet34.yaml \
    "${OUT_DIR}/teacher_12lead_50hz_resnet34_best.pt" \
    teacher_12lead_50hz_resnet34 \
    50

echo ""
echo "── T4: 1-lead 500Hz (SimCLR) ──────────────────"
eval_teacher T4 \
    dafd_mvkt/configs/ta_ii_500hz.yaml \
    "${OUT_DIR}/ta_II_500hz_simclr_seed0_best.pt" \
    ta_II_500hz_simclr_seed0 \
    500

echo ""
echo "── T5: 1-lead 100Hz ───────────────────────────"
eval_teacher T5 \
    dafd_mvkt/configs/ta_ii_500hz.yaml \
    "${OUT_DIR}/ta_II_100hz_from_ta500_seed0_best.pt" \
    ta_II_100hz_from_ta500_seed0 \
    100

echo ""
echo "── T6: 1-lead 50Hz (best student) ────────────"
eval_teacher T6 \
    dafd_mvkt/configs/student_50hz.yaml \
    "${OUT_DIR}/student_II_50hz_bce_ta_mkd_ta_crf_feature_prog_simclr_seed0_best.pt" \
    student_II_50hz_bce_ta_mkd_ta_crf_feature_prog_simclr_seed0 \
    50 \
    "--model_type student"

# ── build summary CSV + MD ─────────────────────────────────────────────────
echo ""
echo "Building teacher bank summary …"
python - << 'PYEOF'
import json, csv, os
from pathlib import Path

TEACHERS = [
    ("T1", "12-lead", 500, "dafd_mvkt/outputs/teacher_best.pt",
     "dafd_mvkt/outputs/metrics_test_teacher.json"),
    ("T2", "12-lead", 100, "dafd_mvkt/outputs/teacher_12lead_100hz_resnet34_best.pt",
     "dafd_mvkt/outputs/metrics_test_teacher_12lead_100hz_resnet34.json"),
    ("T3", "12-lead",  50, "dafd_mvkt/outputs/teacher_12lead_50hz_resnet34_best.pt",
     "dafd_mvkt/outputs/metrics_test_teacher_12lead_50hz_resnet34.json"),
    ("T4", "1-lead",  500, "dafd_mvkt/outputs/ta_II_500hz_simclr_seed0_best.pt",
     "dafd_mvkt/outputs/metrics_test_ta_II_500hz_simclr_seed0.json"),
    ("T5", "1-lead",  100, "dafd_mvkt/outputs/ta_II_100hz_from_ta500_seed0_best.pt",
     "dafd_mvkt/outputs/metrics_test_ta_II_100hz_from_ta500_seed0.json"),
    ("T6", "1-lead",   50,
     "dafd_mvkt/outputs/student_II_50hz_bce_ta_mkd_ta_crf_feature_prog_simclr_seed0_best.pt",
     "dafd_mvkt/outputs/metrics_test_student_II_50hz_bce_ta_mkd_ta_crf_feature_prog_simclr_seed0.json"),
]

BCE_STUDENT_AUC = 0.8061

rows = []
for tid, lead_type, hz, ckpt, metrics_f in TEACHERS:
    ckpt_ok = os.path.exists(ckpt)
    if os.path.exists(metrics_f):
        d = json.load(open(metrics_f))
        auc = d["macro_auc"]
        f1  = d.get("macro_f1_tuned", d.get("macro_f1", 0))
        status = "ready"
        if tid == "T6" and auc <= BCE_STUDENT_AUC:
            status = "WARNING:weak_T6"
    else:
        auc = float("nan"); f1 = float("nan")
        status = "missing_metrics"
    rows.append({
        "teacher_id": tid, "lead_type": lead_type, "hz": hz,
        "checkpoint": ckpt, "ckpt_exists": ckpt_ok,
        "auc": round(auc, 4), "f1_tuned": round(f1, 4), "status": status,
    })

out_dir = Path("dafd_mvkt/outputs/multiteacher")
out_dir.mkdir(exist_ok=True)

csv_path = out_dir / "teacher_bank_summary.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

md_path = out_dir / "teacher_bank_summary.md"
with open(md_path, "w") as f:
    f.write("# Teacher Bank Summary\n\n")
    f.write(f"| ID | Type | Hz | AUC | F1_tuned | Status |\n")
    f.write(f"|----|----|-----|------|---------|--------|\n")
    for r in rows:
        f.write(f"| {r['teacher_id']} | {r['lead_type']} | {r['hz']} | "
                f"{r['auc']} | {r['f1_tuned']} | {r['status']} |\n")
    f.write(f"\nStudent BCE 50Hz baseline: AUC={BCE_STUDENT_AUC}\n")

print(f"Saved: {csv_path}")
print(f"Saved: {md_path}")
for r in rows:
    flag = " ⚠" if "WARNING" in r["status"] else ""
    print(f"  {r['teacher_id']} ({r['lead_type']} {r['hz']}Hz): AUC={r['auc']}  F1={r['f1_tuned']}  [{r['status']}]{flag}")
PYEOF

echo ""
echo "Teacher bank preparation complete."
echo "Summary: dafd_mvkt/outputs/multiteacher/teacher_bank_summary.md"
