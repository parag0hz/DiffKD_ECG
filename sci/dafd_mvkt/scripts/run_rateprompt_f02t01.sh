#!/usr/bin/env bash
# RatePrompt-KD — Sampling-Rate Prompted Knowledge Distillation
# Baseline: F02T01 (T=1.5, w=0.60/0.40, AUC=0.84644)
# Run from /home/kwy00/sci
set -euo pipefail

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}

PY=/home/kwy00/anaconda3/envs/ecg/bin/python
OUT_DIR="dafd_mvkt/outputs/rateprompt_f02t01"

FORKJOIN="dafd_mvkt/outputs/forkjoin"
B500="${FORKJOIN}/F02_branch2_1L500_from_12L100_seed${SEED}_best.pt"
B100="${FORKJOIN}/F02_branch3_1L100_from_12L100_seed${SEED}_best.pt"
SIMCLR="dafd_mvkt/outputs/simclr/simclr_II_50hz_seed${SEED}_encoder.pt"

mkdir -p "${OUT_DIR}/logs"

for ckpt in "${B500}" "${B100}" "${SIMCLR}"; do
    [ -f "${ckpt}" ] || { echo "[ERROR] Missing: ${ckpt}"; exit 1; }
done

run_or_skip() {
    local label="$1"; shift
    local mpath="$1"; shift
    [ -f "${mpath}" ] && { echo "   [skip] ${label}"; return 0; }
    echo ""; echo "   [run]  ${label}"
    "$@" 2>> "${OUT_DIR}/errors.log" || { echo "   [ERROR] ${label}"; return 1; }
}

BASE=(
    --data_dir "${DATA_DIR}" --lead "${LEAD}"
    --teacher_1l500_ckpt "${B500}"
    --teacher_1l100_ckpt "${B100}"
    --student_init_ckpt  "${SIMCLR}"
    --weights "0.60,0.40" --temperature 1.5
    --lambda_kd 1.0
    --seed "${SEED}"
    --output_dir "${OUT_DIR}"
)

run_rp() {
    local VID="$1"; shift
    local MPATH="${OUT_DIR}/metrics_test_student_II_50hz_rp_${VID}_seed${SEED}.json"
    run_or_skip "${VID}" "${MPATH}" \
        ${PY} dafd_mvkt/train_rateprompt.py "${BASE[@]}" \
            --variant_id "${VID}" "$@" \
        | tee "${OUT_DIR}/logs/${VID}_seed${SEED}.log"
}

echo "==== RatePrompt-KD F02T01 (seed=${SEED}) ===="

# RP01: time PE only, 50Hz only (no multi-rate)
#   Hypothesis: explicit time encoding helps even at fixed 50Hz
run_rp RP01 \
    --rate_conditioning time_pe \
    --rate_set "50"

# RP02: FiLM conditioning + multi-rate training (50/100/250/500 Hz)
#   Hypothesis: multi-rate exposure + rate-adaptive feature modulation improves generalization
#   batch_size=128: fs=500 → student [B,1,5000], gradient activations ~4x larger than 50Hz
run_rp RP02 \
    --rate_conditioning fs_film \
    --multi_rate_train \
    --rate_set "50,100,250,500" \
    --batch_size 128

# RP03: time PE + FiLM + multi-rate (50/100/250/500 Hz)
#   Hypothesis: combined PE and FiLM provides maximal rate-awareness
#   batch_size=128: same OOM risk as RP02
run_rp RP03 \
    --rate_conditioning time_pe_film \
    --multi_rate_train \
    --rate_set "50,100,250,500" \
    --batch_size 128

# RP04: time PE + FiLM + multi-rate (50/100/500 Hz, skip 250)
#   Hypothesis: skip interpolated 250Hz; use only natively-stored rates
#   batch_size=128: same OOM risk
run_rp RP04 \
    --rate_conditioning time_pe_film \
    --multi_rate_train \
    --rate_set "50,100,500" \
    --batch_size 128

# ── summary ──────────────────────────────────────────────────────────────────
echo ""
echo "Building summary …"
${PY} - << 'PYEOF'
import json, csv, glob, os

OUT_DIR = "dafd_mvkt/outputs/rateprompt_f02t01"
SEED    = 0
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
BASE_AUC = 0.84644
BASE_F1  = 0.62544

F02_JSON = "dafd_mvkt/outputs/f02_opt/metrics_test_student_II_50hz_f02_F02T01_seed0.json"

def load_row(path):
    d = json.load(open(path))
    row = {
        "variant_id":        d.get("variant_id", "?"),
        "rate_conditioning": d.get("rate_conditioning", "none"),
        "multi_rate":        d.get("multi_rate_train", False),
        "rate_set":          d.get("rate_set", "50"),
        "AUC_macro":         d["macro_auc"],
        "delta_AUC":         round(d["macro_auc"] - BASE_AUC, 5),
        "F1_tuned":          d["macro_f1_tuned"],
        "val_best":          d.get("val_auc_best", ""),
        "seed":              d.get("_meta", {}).get("seed", SEED),
    }
    for c in CLASSES:
        row[f"{c}_AUC"] = d.get(f"auc_{c}", "")
    return row

rows = []
if os.path.exists(F02_JSON):
    d = json.load(open(F02_JSON))
    r = {
        "variant_id": "F02T01", "rate_conditioning": "none",
        "multi_rate": False, "rate_set": "50",
        "AUC_macro": d["macro_auc"], "delta_AUC": 0.0,
        "F1_tuned": d["macro_f1_tuned"], "val_best": d.get("val_auc_best", ""),
        "seed": 0,
    }
    for c in CLASSES:
        r[f"{c}_AUC"] = d.get(f"auc_{c}", "")
    rows.append(r)

for f in sorted(glob.glob(f"{OUT_DIR}/metrics_test_student_II_50hz_rp_RP*_seed{SEED}.json")):
    rows.append(load_row(f))

if not rows:
    print("  No results yet.")
else:
    cols = ["variant_id", "rate_conditioning", "multi_rate", "rate_set",
            "AUC_macro", "delta_AUC", "F1_tuned", "val_best",
            "NORM_AUC", "MI_AUC", "STTC_AUC", "CD_AUC", "HYP_AUC", "seed"]
    with open(f"{OUT_DIR}/summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {OUT_DIR}/summary.csv  ({len(rows)} rows)")

    print()
    print(f"  {'variant':<8} {'cond':<14} {'multi':>5}  {'AUC':>7}  {'ΔAUC':>8}  {'F1':>7}  "
          f"{'HYP':>7}  {'MI':>7}  {'STTC':>7}  {'CD':>7}  {'NORM':>7}")
    print("  " + "-" * 88)
    for r in rows:
        flag = " ★" if r["delta_AUC"] > 0.001 else (" ~" if abs(r["delta_AUC"]) <= 0.0005 else "")
        print(f"  {r['variant_id']:<8} {r['rate_conditioning']:<14} "
              f"{'T' if r['multi_rate'] else 'F':>5}  "
              f"{r['AUC_macro']:>7.4f}  {r['delta_AUC']:>+8.4f}  {r['F1_tuned']:>7.4f}  "
              f"{r.get('HYP_AUC',''):>7}  {r.get('MI_AUC',''):>7}  "
              f"{r.get('STTC_AUC',''):>7}  {r.get('CD_AUC',''):>7}  "
              f"{r.get('NORM_AUC',''):>7}{flag}")
PYEOF

echo ""
echo "==== RatePrompt complete. Summary: ${OUT_DIR}/summary.csv ===="
