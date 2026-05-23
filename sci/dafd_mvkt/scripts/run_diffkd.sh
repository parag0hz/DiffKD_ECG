#!/usr/bin/env bash
# DiffKD experiments runner (odd IDs: this machine RTX5080)
# Run from /home/kwy00/sci
set -euo pipefail

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}

PY=/home/kwy00/anaconda3/envs/ecg/bin/python
OUT_DIR="dafd_mvkt/outputs/diffkd"
F02T01="dafd_mvkt/outputs/f02_opt/student_II_50hz_f02_F02T01_seed0_best.pt"
B500="dafd_mvkt/outputs/forkjoin/F02_branch2_1L500_from_12L100_seed0_best.pt"
B100="dafd_mvkt/outputs/forkjoin/F02_branch3_1L100_from_12L100_seed0_best.pt"

mkdir -p "${OUT_DIR}"

for ckpt in "${F02T01}" "${B500}" "${B100}"; do
    [ -f "${ckpt}" ] || { echo "[ERROR] Missing: ${ckpt}"; exit 1; }
done

run_or_skip() {
    local label="$1"; shift
    local mpath="$1"; shift
    [ -f "${mpath}" ] && { echo "   [skip] ${label}"; return 0; }
    echo ""; echo "   [run]  ${label}"
    "$@"
}

BASE=(
    --config dafd_mvkt/configs/adaptive_50hz.yaml
    --data_dir "${DATA_DIR}"
    --f02t01_ckpt   "${F02T01}"
    --teacher_1l500_ckpt "${B500}"
    --teacher_1l100_ckpt "${B100}"
    --weights "0.60,0.40" --temperature 1.5 --lambda_kd 1.0
    --pred_type v_prediction --feat_layer layer4 --cond_type concat
    --lambda_diff 0.5 --diff_steps 100
    --output_dir "${OUT_DIR}" --seed "${SEED}"
)

# ── DKD01: Naive DiffKD (baseline for this series) ──────────────────────────
run_or_skip DKD01 "${OUT_DIR}/DKD01/metrics_test.json" \
    ${PY} dafd_mvkt/train_diffkd.py "${BASE[@]}" \
        --variant_id DKD01 \
        --phase1_epochs 10 --phase2_epochs 80 --phase3_epochs 10 \
    2>&1 | tee "${OUT_DIR}/DKD01_run.log"

# ── DKD05: layer3 feature ablation ([B,256,32] instead of [B,512,16]) ────────
# Tests if 32-timestep feature space helps vs 16-timestep layer4
run_or_skip DKD05 "${OUT_DIR}/DKD05/metrics_test.json" \
    ${PY} dafd_mvkt/train_diffkd.py "${BASE[@]}" \
        --variant_id DKD05 \
        --feat_layer layer3 \
        --phase1_epochs 10 --phase2_epochs 80 --phase3_epochs 10 \
    2>&1 | tee "${OUT_DIR}/DKD05_run.log"

# ── DKD07: No phase1 pretrain (joint from ep1) ────────────────────────────────
run_or_skip DKD07 "${OUT_DIR}/DKD07/metrics_test.json" \
    ${PY} dafd_mvkt/train_diffkd.py "${BASE[@]}" \
        --variant_id DKD07 \
        --phase2_epochs 100 \
    2>&1 | tee "${OUT_DIR}/DKD07_run.log"

# ── DKD09: epsilon-prediction (ablation vs v-prediction) ─────────────────────
run_or_skip DKD09 "${OUT_DIR}/DKD09/metrics_test.json" \
    ${PY} dafd_mvkt/train_diffkd.py "${BASE[@]}" \
        --variant_id DKD09 \
        --pred_type epsilon \
        --phase1_epochs 10 --phase2_epochs 80 --phase3_epochs 10 \
    2>&1 | tee "${OUT_DIR}/DKD09_run.log"

# ── DKD11_compact: smaller UNet (mults=1,2,2 → ~2.5M) ───────────────────────
# Run only after DKD02 (4090) result confirmed >= 0.847
# Uncomment when ready:
# run_or_skip DKD11_compact "${OUT_DIR}/DKD11_compact/metrics_test.json" \
#     ${PY} dafd_mvkt/train_diffkd.py "${BASE[@]}" \
#         --variant_id DKD11_compact \
#         --unet_base_ch 128 --unet_mults "1,2,2" \
#         --phase1_epochs 10 --phase2_epochs 80 --phase3_epochs 10 \
#     2>&1 | tee "${OUT_DIR}/DKD11_compact_run.log"

# ── DKD03: unconditional ablation — DO NOT RUN until DKD02 (4090) complete ───
# run_or_skip DKD03 "${OUT_DIR}/DKD03/metrics_test.json" \
#     ${PY} dafd_mvkt/train_diffkd.py "${BASE[@]}" \
#         --variant_id DKD03 \
#         --unconditional \
#         --phase1_epochs 10 --phase2_epochs 80 --phase3_epochs 10 \
#     2>&1 | tee "${OUT_DIR}/DKD03_run.log"

# ── summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== DiffKD Summary ==="
${PY} - << 'PYEOF'
import json, glob, os
OUT = "dafd_mvkt/outputs/diffkd"
BASE_AUC = 0.84644
CLASSES = ["NORM","MI","STTC","CD","HYP"]
rows = []
for vdir in sorted(glob.glob(f"{OUT}/DKD*/metrics_test.json")):
    d = json.load(open(vdir))
    rows.append(d)
if not rows:
    print("  No results yet.")
else:
    print(f"  {'variant':<16} {'AUC':>7}  {'ΔAUC':>8}  {'F1t':>7}  {'HYP':>7}  {'MI':>7}")
    print("  " + "-"*60)
    for d in rows:
        auc = d["macro_auc"]; da = auc - BASE_AUC
        flag = " ★" if da > 0.001 else (" ~" if abs(da) <= 0.0005 else "")
        print(f"  {d['variant_id']:<16} {auc:>7.5f}  {da:>+8.5f}  "
              f"{d['macro_f1_tuned']:>7.5f}  "
              f"{d.get('auc_HYP',0):>7.4f}  {d.get('auc_MI',0):>7.4f}{flag}")
PYEOF
echo "======================="
