#!/usr/bin/env bash
# T=1.5 Teacher Weight Tuning (TW15_01~08, skip TW15_03)
# Baseline: F02T01 (w500=0.60, w100=0.40, T=1.5, AUC=0.8464)
#
# Usage:
#   DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#     bash dafd_mvkt/scripts/run_tw15_weight_tune.sh

set -euo pipefail

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}

PY=/home/kwy00/anaconda3/envs/ecg/bin/python
CONFIG="dafd_mvkt/configs/adaptive_50hz.yaml"
FORKJOIN_DIR="dafd_mvkt/outputs/forkjoin"
OUT_DIR="dafd_mvkt/outputs/tw15_weight_tune"

B500="${FORKJOIN_DIR}/F02_branch2_1L500_from_12L100_seed${SEED}_best.pt"
B100="${FORKJOIN_DIR}/F02_branch3_1L100_from_12L100_seed${SEED}_best.pt"
SIMCLR="dafd_mvkt/outputs/simclr/simclr_II_50hz_seed${SEED}_encoder.pt"

CMD_LOG="${OUT_DIR}/tw15_commands.log"
ERR_LOG="${OUT_DIR}/errors.log"

echo "============================================================"
echo "  T=1.5 Teacher Weight Tuning (TW15)"
echo "  DATA_DIR=${DATA_DIR}  LEAD=${LEAD}  SEED=${SEED}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  Baseline: F02T01 (w=0.60,0.40 T=1.5 AUC=0.8464)"
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

# ── dep check ─────────────────────────────────────────────────────────────────
for ckpt in "${B500}" "${B100}" "${SIMCLR}"; do
    if [ ! -f "${ckpt}" ]; then
        echo "[ERROR] Missing: ${ckpt}"; exit 1
    fi
done

run_tw15() {
    local VID="$1"; local W="$2"
    local OUT_NAME="student_II_50hz_f02_${VID}_seed${SEED}"
    run_or_skip "${VID} (w=${W})" "${OUT_DIR}/metrics_test_${OUT_NAME}.json" \
        ${PY} dafd_mvkt/train_f02_student_tune.py \
            --config "${CONFIG}" --data_dir "${DATA_DIR}" --lead "${LEAD}" \
            --teacher_1l500_ckpt "${B500}" --teacher_1l100_ckpt "${B100}" \
            --student_init_ckpt "${SIMCLR}" \
            --weights "${W}" \
            --temperature 1.5 \
            --attention_type none \
            --lambda_att 0.0 \
            --variant_id "${VID}" \
            --output_dir "${OUT_DIR}" \
            --seed "${SEED}"
}

# TW15_03 is skipped (already accounted for in analysis)
run_tw15 TW15_01 "0.50,0.50"
run_tw15 TW15_02 "0.55,0.45"
run_tw15 TW15_04 "0.65,0.35"
run_tw15 TW15_05 "0.70,0.30"
run_tw15 TW15_06 "0.40,0.60"
run_tw15 TW15_07 "0.30,0.70"
run_tw15 TW15_08 "0.75,0.25"

# ── build summary.csv (includes F02T01 as reference) ─────────────────────────
echo ""
echo "Building summary.csv …"
${PY} - << 'PYEOF'
import json, csv, glob, os, sys

OUT_DIR  = "dafd_mvkt/outputs/tw15_weight_tune"
SEED     = 0
CLASSES  = ["NORM", "MI", "STTC", "CD", "HYP"]

# F02T01 reference values (do not rerun)
F02T01_BASE = {
    "variant_id":  "F02T01",
    "w500": 0.600, "w100": 0.400,
    "temperature": 1.5,
    "AUC_macro":   0.8464,
    "F1_tuned":    0.6254,
    "F1_at_0.5":   None,
    "NORM_AUC": 0.9005, "MI_AUC": 0.8217,
    "STTC_AUC": 0.8742, "CD_AUC": 0.8644, "HYP_AUC": 0.7713,
    "NORM_F1":  0.8113, "MI_F1":  0.5918,
    "STTC_F1":  0.6637, "CD_F1":  0.6777, "HYP_F1":  0.3827,
    "best_epoch": None,
    "seed": SEED,
}

rows = []

# load completed TW15 experiments
for f in sorted(glob.glob(f"{OUT_DIR}/metrics_test_student_II_50hz_f02_TW15_*_seed{SEED}.json")):
    d = json.load(open(f))
    row = {
        "variant_id":  d["variant_id"],
        "w500":        round(float(d["w500"]), 3),
        "w100":        round(float(d["w100"]), 3),
        "temperature": d["temperature"],
        "AUC_macro":   d["macro_auc"],
        "F1_tuned":    d["macro_f1_tuned"],
        "F1_at_0.5":   d["macro_f1_0_5"],
        "best_epoch":  d.get("val_auc_best", None),
        "seed":        d.get("_meta", {}).get("seed", SEED),
    }
    for c in CLASSES:
        row[f"{c}_AUC"] = d.get(f"auc_{c}")
        row[f"{c}_F1"]  = d.get(f"f1_tuned_{c}")
    rows.append(row)

# prepend F02T01 reference
f02t1_json = "dafd_mvkt/outputs/f02_opt/metrics_test_student_II_50hz_f02_F02T01_seed0.json"
if os.path.exists(f02t1_json):
    d = json.load(open(f02t1_json))
    ref = {
        "variant_id":  "F02T01",
        "w500":        round(float(d["w500"]), 3),
        "w100":        round(float(d["w100"]), 3),
        "temperature": d["temperature"],
        "AUC_macro":   d["macro_auc"],
        "F1_tuned":    d["macro_f1_tuned"],
        "F1_at_0.5":   d["macro_f1_0_5"],
        "best_epoch":  d.get("val_auc_best", None),
        "seed":        d.get("_meta", {}).get("seed", SEED),
    }
    for c in CLASSES:
        ref[f"{c}_AUC"] = d.get(f"auc_{c}")
        ref[f"{c}_F1"]  = d.get(f"f1_tuned_{c}")
else:
    ref = F02T01_BASE

all_rows = [ref] + rows

cols = ["variant_id", "w500", "w100", "temperature",
        "AUC_macro", "F1_tuned", "F1_at_0.5",
        "NORM_AUC", "MI_AUC", "STTC_AUC", "CD_AUC", "HYP_AUC",
        "NORM_F1",  "MI_F1",  "STTC_F1",  "CD_F1",  "HYP_F1",
        "best_epoch", "seed"]

csv_path = f"{OUT_DIR}/summary.csv"
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(all_rows)
print(f"  Saved: {csv_path}  ({len(all_rows)} rows)")
PYEOF

# ── print results ─────────────────────────────────────────────────────────────
echo ""
${PY} - << 'PYEOF'
import json, glob

OUT_DIR = "dafd_mvkt/outputs/tw15_weight_tune"
SEED    = 0
CLASSES = ["NORM","MI","STTC","CD","HYP"]
BASE_AUC = 0.8464; BASE_F1 = 0.6254

rows = []

# F02T01 reference
f02t1 = "dafd_mvkt/outputs/f02_opt/metrics_test_student_II_50hz_f02_F02T01_seed0.json"
import os
if os.path.exists(f02t1):
    d = json.load(open(f02t1))
    rows.append(("F02T01 (base)", d["w500"], d["w100"],
                 d["macro_auc"], d["macro_f1_tuned"],
                 [d.get(f"auc_{c}",0) for c in CLASSES],
                 [d.get(f"f1_tuned_{c}",0) for c in CLASSES]))
else:
    rows.append(("F02T01 (base)", 0.600, 0.400, BASE_AUC, BASE_F1,
                 [0.9005,0.8217,0.8742,0.8644,0.7713],
                 [0.8113,0.5918,0.6637,0.6777,0.3827]))

for f in sorted(glob.glob(f"{OUT_DIR}/metrics_test_student_II_50hz_f02_TW15_*_seed{SEED}.json")):
    d = json.load(open(f))
    rows.append((d["variant_id"], d["w500"], d["w100"],
                 d["macro_auc"], d["macro_f1_tuned"],
                 [d.get(f"auc_{c}",0) for c in CLASSES],
                 [d.get(f"f1_tuned_{c}",0) for c in CLASSES]))

# sort by AUC desc
rows.sort(key=lambda r: r[3], reverse=True)

print("=" * 75)
print("  T=1.5 Weight Tuning Results (AUC 순위)")
print("=" * 75)
print(f"  {'variant':<14} {'w500':>5} {'w100':>5}  {'AUC':>7}  {'F1t':>7}  {'ΔAUC':>7}  {'ΔF1':>7}")
print("  " + "-" * 63)
for vid, w5, w1, auc, f1, _, _ in rows:
    da = auc - BASE_AUC; df = f1 - BASE_F1
    mark = " ★" if auc > BASE_AUC else ""
    print(f"  {vid:<14} {w5:>5.3f} {w1:>5.3f}  {auc:>7.4f}  {f1:>7.4f}  {da:>+7.4f}  {df:>+7.4f}{mark}")

print()
print("  Per-class AUC")
print(f"  {'variant':<14} " + "  ".join(f"{c:>7}" for c in CLASSES))
print("  " + "-" * 55)
for vid, w5, w1, auc, f1, aucs, _ in rows:
    print(f"  {vid:<14} " + "  ".join(f"{a:>7.4f}" for a in aucs))

print()
print("  Per-class F1_tuned")
print(f"  {'variant':<14} " + "  ".join(f"{c:>7}" for c in CLASSES))
print("  " + "-" * 55)
for vid, w5, w1, auc, f1, _, f1s in rows:
    print(f"  {vid:<14} " + "  ".join(f"{v:>7.4f}" for v in f1s))
PYEOF

echo ""
echo "============================================================"
echo "  TW15 weight tuning complete."
echo "  Summary: ${OUT_DIR}/summary.csv"
echo "============================================================"
