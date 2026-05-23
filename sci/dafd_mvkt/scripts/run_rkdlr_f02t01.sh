#!/usr/bin/env bash
# RKDLR — Recoverability-aware KD for Low-Rate ECG
# Baseline: F02T01 (T=1.5, w=0.60/0.40, AUC=0.8464)
# Run from /home/kwy00/sci
set -euo pipefail

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}

PY=/home/kwy00/anaconda3/envs/ecg/bin/python
CONFIG="dafd_mvkt/configs/adaptive_50hz.yaml"
FORKJOIN="dafd_mvkt/outputs/forkjoin"
OUT_DIR="dafd_mvkt/outputs/rkdlr_f02t01"

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
    --config "${CONFIG}" --data_dir "${DATA_DIR}" --lead "${LEAD}"
    --teacher_1l500_ckpt "${B500}" --teacher_1l100_ckpt "${B100}"
    --student_init_ckpt  "${SIMCLR}"
    --weights "0.60,0.40" --temperature 1.5
    --attention_type none --lambda_att 0.0
    --method_mode rkdlr
    --output_dir "${OUT_DIR}" --seed "${SEED}"
)

run_rkdlr() {
    local VID="$1"; shift
    local OUT_NAME="student_II_50hz_f02_${VID}_seed${SEED}"
    run_or_skip "${VID}" "${OUT_DIR}/metrics_test_${OUT_NAME}.json" \
        ${PY} dafd_mvkt/train_f02_student_tune.py "${BASE[@]}" \
            --variant_id "${VID}" "$@"
}

echo "==== RKDLR F02T01 (seed=${SEED}) ===="

# Phase gate_full
run_rkdlr RKDLR01 --rkdlr_mode gate_full  --rkdlr_gamma 1.0 --rkdlr_min_weight 0.0
run_rkdlr RKDLR02 --rkdlr_mode gate_full  --rkdlr_gamma 0.5 --rkdlr_min_weight 0.0
run_rkdlr RKDLR03 --rkdlr_mode gate_full  --rkdlr_gamma 2.0 --rkdlr_min_weight 0.0
run_rkdlr RKDLR04 --rkdlr_mode gate_full  --rkdlr_gamma 1.0 --rkdlr_min_weight 0.25
run_rkdlr RKDLR05 --rkdlr_mode degraded_target
run_rkdlr RKDLR06 --rkdlr_mode mixed_target --rkdlr_gamma 1.0 --rkdlr_min_weight 0.0

# summary
echo ""
echo "Building summary.csv …"
${PY} - << 'PYEOF'
import json, csv, glob, os

OUT_DIR = "dafd_mvkt/outputs/rkdlr_f02t01"
SEED    = 0
CLASSES = ["NORM","MI","STTC","CD","HYP"]

F02_JSON = "dafd_mvkt/outputs/f02_opt/metrics_test_student_II_50hz_f02_F02T01_seed0.json"
BASE_AUC = 0.8464; BASE_F1 = 0.6254

def load_row(path):
    d = json.load(open(path))
    vid = d.get("variant_id","?")
    row = {
        "variant_id":          vid,
        "method_mode":         d.get("method_mode","base"),
        "rkdlr_mode":          d.get("rkdlr_mode",""),
        "rkdlr_gamma":         d.get("rkdlr_gamma",""),
        "rkdlr_min_weight":    d.get("rkdlr_min_weight",""),
        "AUC_macro":           d["macro_auc"],
        "F1_tuned":            d["macro_f1_tuned"],
        "F1_at_0.5":           d.get("macro_f1_0_5",""),
        "mean_recoverability": d.get("mean_recoverability",""),
        "mean_r500":           d.get("mean_r500",""),
        "mean_r100":           d.get("mean_r100",""),
        "best_epoch":          d.get("val_auc_best",""),
        "seed":                d.get("_meta",{}).get("seed", SEED),
        "notes":               "",
    }
    for c in CLASSES:
        row[f"{c}_AUC"] = d.get(f"auc_{c}","")
        row[f"{c}_F1"]  = d.get(f"f1_tuned_{c}","")
    return row

rows = []
if os.path.exists(F02_JSON):
    r = load_row(F02_JSON)
    r["variant_id"]="F02T01"; r["method_mode"]="base"; r["rkdlr_mode"]=""; r["notes"]="baseline"
    rows.append(r)

for f in sorted(glob.glob(f"{OUT_DIR}/metrics_test_student_II_50hz_f02_RKDLR*_seed{SEED}.json")):
    rows.append(load_row(f))

cols = ["variant_id","method_mode","rkdlr_mode","rkdlr_gamma","rkdlr_min_weight",
        "AUC_macro","F1_tuned","F1_at_0.5",
        "NORM_AUC","MI_AUC","STTC_AUC","CD_AUC","HYP_AUC",
        "NORM_F1","MI_F1","STTC_F1","CD_F1","HYP_F1",
        "mean_recoverability","mean_r500","mean_r100",
        "best_epoch","seed","notes"]
with open(f"{OUT_DIR}/summary.csv","w",newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print(f"  Saved: {OUT_DIR}/summary.csv  ({len(rows)} rows)")
PYEOF

# print table
${PY} - << 'PYEOF'
import json, glob, os
OUT_DIR="dafd_mvkt/outputs/rkdlr_f02t01"; SEED=0
CLASSES=["NORM","MI","STTC","CD","HYP"]; BASE_AUC=0.8464; BASE_F1=0.6254
rows=[]
f02="dafd_mvkt/outputs/f02_opt/metrics_test_student_II_50hz_f02_F02T01_seed0.json"
if os.path.exists(f02):
    d=json.load(open(f02))
    rows.append(("F02T01","base","",d["macro_auc"],d["macro_f1_tuned"],[d.get(f"auc_{c}",0)for c in CLASSES],None,None,None))
for f in sorted(glob.glob(f"{OUT_DIR}/metrics_test_student_II_50hz_f02_RKDLR*_seed{SEED}.json")):
    d=json.load(open(f))
    rows.append((d["variant_id"],d.get("method_mode","?"),d.get("rkdlr_mode","?"),
                 d["macro_auc"],d["macro_f1_tuned"],[d.get(f"auc_{c}",0)for c in CLASSES],
                 d.get("mean_recoverability"),d.get("mean_r500"),d.get("mean_r100")))
rows.sort(key=lambda r:r[3],reverse=True)
print("="*90)
print(f"  {'variant':<10} {'mode':<16} {'AUC':>7}  {'ΔAUC':>8}  {'F1':>7}  {'HYP':>7}  {'MI':>7}  {'r':>6}")
print("  "+"-"*78)
for vid,mm,rm,auc,f1,pcs,r,r5,r1 in rows:
    da=auc-BASE_AUC; flag=" ★"if auc>BASE_AUC+0.001 else(" ~"if abs(da)<=0.0005 else"")
    rs=f"{r:.3f}"if r is not None else"  -  "
    print(f"  {vid:<10} {rm:<16} {auc:>7.4f}  {da:>+8.4f}  {f1:>7.4f}  {pcs[4]:>7.4f}  {pcs[1]:>7.4f}  {rs}{flag}")
PYEOF

echo ""
echo "==== RKDLR complete. Summary: ${OUT_DIR}/summary.csv ===="
