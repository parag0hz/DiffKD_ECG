#!/usr/bin/env bash
# Adaptive Knowledge Distillation on F02T01 baseline
# Baseline: F02T01 (w500=0.60, w100=0.40, T=1.5, AUC=0.8464, F1_tuned=0.6254)
#
# Phases:
#   A: AKD01-06  — confidence / agreement / confidence_agreement
#   B: EKD01-02  — ensemble_prob / ensemble_logit
#   C: LCKD01    — label_correlation
#   D: DKD01-02  — dynamic_weight
#   E: CRD01     — class_reliability (uses teacher AUC weights)
#
# Usage:
#   DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#     bash dafd_mvkt/scripts/run_adaptive_kd_f02t01.sh
#
#   # Phase A+B only (default)
#   bash dafd_mvkt/scripts/run_adaptive_kd_f02t01.sh
#
#   # All phases
#   RUN_ALL=1 bash dafd_mvkt/scripts/run_adaptive_kd_f02t01.sh

set -euo pipefail

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}
RUN_ALL=${RUN_ALL:-0}

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
echo "  Adaptive KD (AKD) — F02T01 baseline"
echo "  DATA_DIR=${DATA_DIR}  LEAD=${LEAD}  SEED=${SEED}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  Baseline: F02T01 (w=0.60,0.40 T=1.5 AUC=0.8464)"
echo "  RUN_ALL=${RUN_ALL}  (0=PhaseA+B only, 1=all phases)"
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

# ── Phase A: confidence / agreement / confidence_agreement ────────────────────
echo ""
echo "── Phase A: Adaptive Weights (confidence / agreement) ──"

run_akd AKD01 \
    --akd_mode confidence \
    --akd_gamma 1.0 \
    --akd_detach_weight

run_akd AKD02 \
    --akd_mode agreement \
    --akd_gamma 1.0 \
    --akd_detach_weight

run_akd AKD03 \
    --akd_mode confidence_agreement \
    --akd_gamma 1.0 \
    --akd_detach_weight

run_akd AKD04 \
    --akd_mode confidence_agreement \
    --akd_gamma 0.5 \
    --akd_detach_weight

run_akd AKD05 \
    --akd_mode confidence_agreement \
    --akd_gamma 2.0 \
    --akd_detach_weight

run_akd AKD06 \
    --akd_mode confidence_agreement \
    --akd_gamma 1.0 \
    --akd_min_weight 0.25 \
    --akd_detach_weight

# ── Phase B: ensemble ─────────────────────────────────────────────────────────
echo ""
echo "── Phase B: Ensemble KD ──"

run_akd EKD01 \
    --akd_mode ensemble_prob

run_akd EKD02 \
    --akd_mode ensemble_logit

# ── Phase C-E: optional (RUN_ALL=1) ──────────────────────────────────────────
if [ "${RUN_ALL}" = "1" ]; then

    echo ""
    echo "── Phase C: Label Correlation KD ──"

    run_akd LCKD01 \
        --akd_mode label_correlation \
        --label_corr_weight 0.01

    echo ""
    echo "── Phase D: Dynamic Weight ──"

    run_akd DKD01 \
        --akd_mode dynamic_weight \
        --dynamic_w500_start 0.50 \
        --dynamic_w500_end 0.60

    run_akd DKD02 \
        --akd_mode dynamic_weight \
        --dynamic_w500_start 0.55 \
        --dynamic_w500_end 0.65

    echo ""
    echo "── Phase E: Class Reliability ──"

    # NORM=0.9005, MI=0.8217, STTC=0.8742, CD=0.8644, HYP=0.7713 (F02T01 per-class AUC)
    # normalized to sum=1: 0.9005+0.8217+0.8742+0.8644+0.7713 = 4.2321
    run_akd CRD01 \
        --akd_mode class_reliability \
        --class_reliability_source manual \
        --class_reliability_values "0.9005,0.8217,0.8742,0.8644,0.7713"

fi

# ── build summary.csv ─────────────────────────────────────────────────────────
echo ""
echo "Building summary.csv …"
${PY} - << 'PYEOF'
import json, csv, glob, os

OUT_DIR  = "dafd_mvkt/outputs/adaptive_kd_f02t01"
SEED     = 0
CLASSES  = ["NORM", "MI", "STTC", "CD", "HYP"]

# F02T01 reference
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
    "best_epoch": None, "seed": SEED,
}

def load_row(path):
    d = json.load(open(path))
    row = {
        "variant_id":       d.get("variant_id", "?"),
        "akd_mode":         d.get("akd_mode", "base"),
        "w500":             round(float(d["w500"]), 3),
        "w100":             round(float(d["w100"]), 3),
        "temperature":      d["temperature"],
        "AUC_macro":        d["macro_auc"],
        "F1_tuned":         d["macro_f1_tuned"],
        "F1_at_0.5":        d.get("macro_f1_0_5"),
        "akd_gamma":        d.get("akd_gamma"),
        "akd_min_weight":   d.get("akd_min_weight"),
        "label_corr_weight":d.get("label_corr_weight"),
        "best_epoch":       d.get("val_auc_best"),
        "seed":             d.get("_meta", {}).get("seed", SEED),
    }
    for c in CLASSES:
        row[f"{c}_AUC"] = d.get(f"auc_{c}")
        row[f"{c}_F1"]  = d.get(f"f1_tuned_{c}")
    return row

rows = []
if os.path.exists(F02T01_JSON):
    rows.append(load_row(F02T01_JSON))
    rows[0]["variant_id"] = "F02T01"
    rows[0]["akd_mode"] = "base"
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
    "best_epoch", "seed",
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

rows = []
f02t1 = "dafd_mvkt/outputs/f02_opt/metrics_test_student_II_50hz_f02_F02T01_seed0.json"
if os.path.exists(f02t1):
    d = json.load(open(f02t1))
    rows.append(("F02T01 (base)", d.get("akd_mode","base"),
                 d["w500"], d["w100"], d["macro_auc"], d["macro_f1_tuned"],
                 [d.get(f"auc_{c}", 0) for c in CLASSES],
                 [d.get(f"f1_tuned_{c}", 0) for c in CLASSES]))
else:
    rows.append(("F02T01 (base)", "base",
                 0.600, 0.400, BASE_AUC, BASE_F1,
                 [0.9005,0.8217,0.8742,0.8644,0.7713],
                 [0.8113,0.5918,0.6637,0.6777,0.3827]))

for f in sorted(glob.glob(f"{OUT_DIR}/metrics_test_student_II_50hz_f02_*_seed{SEED}.json")):
    d = json.load(open(f))
    rows.append((d["variant_id"], d.get("akd_mode","?"),
                 d["w500"], d["w100"], d["macro_auc"], d["macro_f1_tuned"],
                 [d.get(f"auc_{c}", 0) for c in CLASSES],
                 [d.get(f"f1_tuned_{c}", 0) for c in CLASSES]))

rows.sort(key=lambda r: r[4], reverse=True)

print("=" * 80)
print("  Adaptive KD Results (AUC 순위)")
print("=" * 80)
print(f"  {'variant':<12} {'mode':<22} {'AUC':>7}  {'F1t':>7}  {'ΔAUC':>7}  {'ΔF1':>7}")
print("  " + "-" * 68)
for vid, mode, w5, w1, auc, f1, _, _ in rows:
    da = auc - BASE_AUC; df = f1 - BASE_F1
    mark = " ★" if auc > BASE_AUC + 0.001 else (" ~" if abs(da) <= 0.0005 else "")
    print(f"  {vid:<12} {mode:<22} {auc:>7.4f}  {f1:>7.4f}  {da:>+7.4f}  {df:>+7.4f}{mark}")

print()
print("  Per-class AUC")
print(f"  {'variant':<12} " + "  ".join(f"{c:>7}" for c in CLASSES))
print("  " + "-" * 55)
for vid, mode, w5, w1, auc, f1, aucs, _ in rows:
    print(f"  {vid:<12} " + "  ".join(f"{a:>7.4f}" for a in aucs))
PYEOF

echo ""
echo "============================================================"
echo "  Adaptive KD complete."
echo "  Summary: ${OUT_DIR}/summary.csv"
echo "============================================================"
