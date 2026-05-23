#!/usr/bin/env bash
# Adaptive KD Phase C–E: Dynamic Weight / Label Correlation / Class Reliability
# Context: Phase A+B (confidence/agreement/ensemble) all failed vs F02T01 (AUC=0.8464)
# Phase A+B is closed. This script runs the three remaining directions.
#
# Phase C: DKD01-03 — epoch-wise w500 schedule (dynamic_weight)
# Phase D: LCKD01-03 — label correlation auxiliary loss (label_correlation)
# Phase E: CKD01-03  — class-reliability weighting (class_reliability)
#
# Usage (from /home/kwy00/sci):
#   DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#     bash dafd_mvkt/scripts/run_adaptive_kd_phase_cde.sh

set -euo pipefail

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}

PY=/home/kwy00/anaconda3/envs/ecg/bin/python
CONFIG="dafd_mvkt/configs/adaptive_50hz.yaml"
FORKJOIN_DIR="dafd_mvkt/outputs/forkjoin"
OUT_DIR="dafd_mvkt/outputs/adaptive_kd_f02t01"

B500="${FORKJOIN_DIR}/F02_branch2_1L500_from_12L100_seed${SEED}_best.pt"
B100="${FORKJOIN_DIR}/F02_branch3_1L100_from_12L100_seed${SEED}_best.pt"
SIMCLR="dafd_mvkt/outputs/simclr/simclr_II_50hz_seed${SEED}_encoder.pt"

CMD_LOG="${OUT_DIR}/akd_commands.log"
ERR_LOG="${OUT_DIR}/errors.log"

echo "============================================================"
echo "  Adaptive KD Phase C-E (DKD/LCKD/CKD)"
echo "  DATA_DIR=${DATA_DIR}  LEAD=${LEAD}  SEED=${SEED}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  Baseline: F02T01 (AUC=0.8464)  [Phase A+B: STOPPED]"
echo "============================================================"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"

log_cmd() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${CMD_LOG}"; }

run_or_skip() {
    local label="$1"; shift
    local metrics_path="$1"; shift
    if [ -f "${metrics_path}" ]; then
        echo "   [skip] ${label}"
        return 0
    fi
    echo ""
    echo "   [run]  ${label}"
    log_cmd "${label}: $*"
    if ! "$@" 2>> "${ERR_LOG}"; then
        echo "   [ERROR] ${label} failed — see ${ERR_LOG}"
        return 1
    fi
}

for ckpt in "${B500}" "${B100}" "${SIMCLR}"; do
    if [ ! -f "${ckpt}" ]; then
        echo "[ERROR] Missing: ${ckpt}"; exit 1
    fi
done

BASE_ARGS=(
    --config   "${CONFIG}"
    --data_dir "${DATA_DIR}"
    --lead     "${LEAD}"
    --teacher_1l500_ckpt "${B500}"
    --teacher_1l100_ckpt "${B100}"
    --student_init_ckpt  "${SIMCLR}"
    --weights  "0.60,0.40"
    --temperature 1.5
    --attention_type none
    --lambda_att 0.0
    --output_dir "${OUT_DIR}"
    --seed "${SEED}"
)

run_akd() {
    local VID="$1"; shift
    local OUT_NAME="student_II_50hz_f02_${VID}_seed${SEED}"
    run_or_skip "${VID}" "${OUT_DIR}/metrics_test_${OUT_NAME}.json" \
        ${PY} dafd_mvkt/train_f02_student_tune.py \
            "${BASE_ARGS[@]}" \
            --variant_id "${VID}" \
            "$@"
}

# ── Phase C: Dynamic Weight ───────────────────────────────────────────────────
echo ""
echo "── Phase C: Dynamic Weight KD (DKD01-03) ──"
echo "   Epoch-wise w500 schedule; no per-sample gates"

run_akd DKD01 \
    --akd_mode dynamic_weight \
    --dynamic_w500_start 0.50 \
    --dynamic_w500_end   0.60

run_akd DKD02 \
    --akd_mode dynamic_weight \
    --dynamic_w500_start 0.45 \
    --dynamic_w500_end   0.65

run_akd DKD03 \
    --akd_mode dynamic_weight \
    --dynamic_w500_start 0.55 \
    --dynamic_w500_end   0.65

# ── Phase D: Label Correlation KD ─────────────────────────────────────────────
echo ""
echo "── Phase D: Label Correlation KD (LCKD01-03) ──"
echo "   Auxiliary covariance loss; base KD unchanged"

run_akd LCKD01 \
    --akd_mode label_correlation \
    --label_corr_weight 0.001

run_akd LCKD02 \
    --akd_mode label_correlation \
    --label_corr_weight 0.003

run_akd LCKD03 \
    --akd_mode label_correlation \
    --label_corr_weight 0.005

# ── Phase E: Class Reliability KD ─────────────────────────────────────────────
echo ""
echo "── Phase E: Class Reliability KD (CKD01-03) ──"
echo "   Down-weight HYP (and optionally MI) teacher signal"

# NORM=1.0, MI=1.0, STTC=1.0, CD=1.0, HYP=0.75
run_akd CKD01 \
    --akd_mode class_reliability \
    --class_reliability_source manual \
    --class_reliability_values "1.0,1.0,1.0,1.0,0.75"

# NORM=1.0, MI=1.0, STTC=1.0, CD=1.0, HYP=0.50
run_akd CKD02 \
    --akd_mode class_reliability \
    --class_reliability_source manual \
    --class_reliability_values "1.0,1.0,1.0,1.0,0.50"

# NORM=1.0, MI=0.95, STTC=1.0, CD=1.0, HYP=0.75
run_akd CKD03 \
    --akd_mode class_reliability \
    --class_reliability_source manual \
    --class_reliability_values "1.0,0.95,1.0,1.0,0.75"

# ── rebuild summary.csv (all phases) ──────────────────────────────────────────
echo ""
echo "Rebuilding summary.csv …"
${PY} - << 'PYEOF'
import json, csv, glob, os

OUT_DIR  = "dafd_mvkt/outputs/adaptive_kd_f02t01"
SEED     = 0
CLASSES  = ["NORM", "MI", "STTC", "CD", "HYP"]

PHASE_NOTES = {
    "AKD01": "failed — confidence weighting",
    "AKD02": "failed — agreement weighting",
    "AKD03": "failed — conf×agree γ=1.0",
    "AKD04": "failed — conf×agree γ=0.5",
    "AKD05": "failed — conf×agree γ=2.0",
    "AKD06": "failed — conf×agree min_w=0.25",
    "EKD01": "failed — ensemble_prob",
    "EKD02": "failed — ensemble_logit",
}

F02T01_JSON = "dafd_mvkt/outputs/f02_opt/metrics_test_student_II_50hz_f02_F02T01_seed0.json"
F02T01_BASE = {
    "variant_id": "F02T01", "akd_mode": "base",
    "w500": 0.600, "w100": 0.400, "temperature": 1.5,
    "AUC_macro": 0.8464, "F1_tuned": 0.6254, "F1_at_0.5": None,
    "NORM_AUC": 0.9005, "MI_AUC": 0.8217, "STTC_AUC": 0.8742,
    "CD_AUC": 0.8644, "HYP_AUC": 0.7713,
    "NORM_F1": 0.8113, "MI_F1": 0.5918, "STTC_F1": 0.6637,
    "CD_F1": 0.6777, "HYP_F1": 0.3827,
    "akd_gamma": None, "akd_min_weight": None, "label_corr_weight": None,
    "best_epoch": None, "seed": SEED, "notes": "baseline",
}

def load_row(path):
    d = json.load(open(path))
    vid = d.get("variant_id", "?")
    row = {
        "variant_id":        vid,
        "akd_mode":          d.get("akd_mode", "base"),
        "w500":              round(float(d["w500"]), 3),
        "w100":              round(float(d["w100"]), 3),
        "temperature":       d["temperature"],
        "AUC_macro":         round(d["macro_auc"], 5),
        "F1_tuned":          round(d["macro_f1_tuned"], 5),
        "F1_at_0.5":         round(d.get("macro_f1_0_5", 0) or 0, 5),
        "akd_gamma":         d.get("akd_gamma"),
        "akd_min_weight":    d.get("akd_min_weight"),
        "label_corr_weight": d.get("label_corr_weight"),
        "best_epoch":        d.get("val_auc_best"),
        "seed":              d.get("_meta", {}).get("seed", SEED),
        "notes":             PHASE_NOTES.get(vid, ""),
    }
    for c in CLASSES:
        row[f"{c}_AUC"] = round(d.get(f"auc_{c}") or 0, 5)
        row[f"{c}_F1"]  = round(d.get(f"f1_tuned_{c}") or 0, 5)
    return row

rows = []
if os.path.exists(F02T01_JSON):
    ref = load_row(F02T01_JSON)
    ref["variant_id"] = "F02T01"; ref["akd_mode"] = "base"; ref["notes"] = "baseline"
    rows.append(ref)
else:
    rows.append(F02T01_BASE)

for f in sorted(glob.glob(f"{OUT_DIR}/metrics_test_student_II_50hz_f02_*_seed{SEED}.json")):
    rows.append(load_row(f))

cols = [
    "variant_id", "akd_mode", "w500", "w100", "temperature",
    "AUC_macro", "F1_tuned", "F1_at_0.5",
    "NORM_AUC", "MI_AUC", "STTC_AUC", "CD_AUC", "HYP_AUC",
    "NORM_F1",  "MI_F1",  "STTC_F1",  "CD_F1",  "HYP_F1",
    "akd_gamma", "akd_min_weight", "label_corr_weight",
    "best_epoch", "seed", "notes",
]

csv_path = f"{OUT_DIR}/summary.csv"
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"  Saved: {csv_path}  ({len(rows)} rows)")
PYEOF

# ── print results ─────────────────────────────────────────────────────────────
echo ""
${PY} - << 'PYEOF'
import json, glob, os

OUT_DIR  = "dafd_mvkt/outputs/adaptive_kd_f02t01"
SEED     = 0
CLASSES  = ["NORM", "MI", "STTC", "CD", "HYP"]
BASE_AUC = 0.8464; BASE_F1 = 0.6254
BASE_HYP = 0.7713; BASE_MI = 0.8217

rows = []
f02t1 = "dafd_mvkt/outputs/f02_opt/metrics_test_student_II_50hz_f02_F02T01_seed0.json"
if os.path.exists(f02t1):
    d = json.load(open(f02t1))
    rows.append(("F02T01", "base", d["macro_auc"], d["macro_f1_tuned"],
                 [d.get(f"auc_{c}", 0) for c in CLASSES],
                 [d.get(f"f1_tuned_{c}", 0) for c in CLASSES]))
else:
    rows.append(("F02T01", "base", BASE_AUC, BASE_F1,
                 [0.9005,0.8217,0.8742,0.8644,0.7713],
                 [0.8113,0.5918,0.6637,0.6777,0.3827]))

for f in sorted(glob.glob(f"{OUT_DIR}/metrics_test_student_II_50hz_f02_*_seed{SEED}.json")):
    d = json.load(open(f))
    rows.append((d["variant_id"], d.get("akd_mode","?"),
                 d["macro_auc"], d["macro_f1_tuned"],
                 [d.get(f"auc_{c}", 0) for c in CLASSES],
                 [d.get(f"f1_tuned_{c}", 0) for c in CLASSES]))

rows.sort(key=lambda r: r[2], reverse=True)

print("=" * 85)
print("  Adaptive KD Phase A–E Results (AUC 순위)")
print("=" * 85)
print(f"  {'variant':<10} {'mode':<20} {'AUC':>7}  {'F1t':>7}  {'ΔAUC':>8}  {'ΔHYP':>7}  {'ΔMI':>7}")
print("  " + "-" * 73)
for vid, mode, auc, f1, aucs, f1s in rows:
    da   = auc - BASE_AUC
    dhyp = aucs[4] - BASE_HYP
    dmi  = aucs[1] - BASE_MI
    mark = " ★" if auc > BASE_AUC + 0.001 else (" ~" if abs(da) <= 0.0005 else "")
    print(f"  {vid:<10} {mode:<20} {auc:>7.4f}  {f1:>7.4f}  {da:>+8.4f}  {dhyp:>+7.4f}  {dmi:>+7.4f}{mark}")

print()
print("  Per-class AUC")
print(f"  {'variant':<10} " + "  ".join(f"{c:>7}" for c in CLASSES))
print("  " + "-" * 55)
for vid, mode, auc, f1, aucs, _ in rows:
    print(f"  {vid:<10} " + "  ".join(f"{a:>7.4f}" for a in aucs))
PYEOF

echo ""
echo "============================================================"
echo "  Phase C-E complete. See README_AKD.md for judgment."
echo "  Summary: ${OUT_DIR}/summary.csv"
echo "============================================================"
