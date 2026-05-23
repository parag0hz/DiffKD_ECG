#!/usr/bin/env bash
# TG-MTA Attention Experiments (ATTN00–ATTN10)
# Built on top of F02T01 best baseline (w500=0.60, w100=0.40, T=1.5).
#
# Run ATTN03–ATTN07 first (core TG-MTA grid), then optionally all:
#
#   # first batch (default):
#   DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#     bash dafd_mvkt/scripts/run_tgmta_f02.sh
#
#   # all experiments:
#   RUN_ALL=1 DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#     bash dafd_mvkt/scripts/run_tgmta_f02.sh

set -euo pipefail

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}
RUN_ALL=${RUN_ALL:-0}

PY=/home/kwy00/anaconda3/envs/ecg/bin/python
CONFIG="dafd_mvkt/configs/adaptive_50hz.yaml"
FORKJOIN_DIR="dafd_mvkt/outputs/forkjoin"
OUT_DIR="dafd_mvkt/outputs/tgmta_f02"

B500="${FORKJOIN_DIR}/F02_branch2_1L500_from_12L100_seed${SEED}_best.pt"
B100="${FORKJOIN_DIR}/F02_branch3_1L100_from_12L100_seed${SEED}_best.pt"
SIMCLR="dafd_mvkt/outputs/simclr/simclr_II_50hz_seed${SEED}_encoder.pt"

# Best settings from F02 optimization (F02T01 best)
BEST_W="0.60,0.40"
BEST_T="1.5"

CMD_LOG="${OUT_DIR}/tgmta_commands.log"
ERR_LOG="${OUT_DIR}/errors.log"

echo "============================================================"
echo "  TG-MTA Attention Experiments (ATTN00–ATTN10)"
echo "  DATA_DIR=${DATA_DIR}  LEAD=${LEAD}  SEED=${SEED}"
echo "  baseline: w500/w100=${BEST_W}  T=${BEST_T}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  RUN_ALL=${RUN_ALL}"
echo "============================================================"
echo ""

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"

log_cmd() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${CMD_LOG}"; }

run_or_skip() {
    local label="$1"; shift
    local metrics_path="$1"; shift
    if [ -f "${metrics_path}" ]; then
        echo "   [skip] ${label}  (${metrics_path})"
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

# ── check branch checkpoints ──────────────────────────────────────────────────
echo "Checking F02 branch checkpoints …"
for ckpt in "${B500}" "${B100}"; do
    if [ ! -f "${ckpt}" ]; then
        echo "[ERROR] Missing checkpoint: ${ckpt}"
        echo "  Run: DATA_DIR=${DATA_DIR} SEED=${SEED} LEAD=${LEAD} bash dafd_mvkt/scripts/run_fork_join_6.sh"
        exit 1
    fi
done
echo "  1L500: ${B500}"
echo "  1L100: ${B100}"

if [ ! -f "${SIMCLR}" ]; then
    echo "[WARN] SimCLR init not found: ${SIMCLR} — using random init"
fi

# ── helper ───────────────────────────────────────────────────────────────────
run_attn() {
    local VID="$1"; shift
    local OUT_NAME="student_II_50hz_f02_${VID}_seed${SEED}"
    run_or_skip "${VID}" "${OUT_DIR}/metrics_test_${OUT_NAME}.json" \
        ${PY} dafd_mvkt/train_f02_student_tune.py \
            --config "${CONFIG}" --data_dir "${DATA_DIR}" --lead "${LEAD}" \
            --teacher_1l500_ckpt "${B500}" --teacher_1l100_ckpt "${B100}" \
            --student_init_ckpt "${SIMCLR}" \
            --weights "${BEST_W}" --temperature "${BEST_T}" \
            --variant_id "${VID}" --output_dir "${OUT_DIR}" --seed "${SEED}" \
            "$@"
}

# ── optional: ATTN00 + ATTN01 + ATTN02 (run only when RUN_ALL=1) ──────────────
if [ "${RUN_ALL}" = "1" ]; then
    echo "Stage A: Baselines & Module Comparison (ATTN00–ATTN02)"
    echo "─────────────────────────────────────────────────────"

    # ATTN00: no attention (F02T01 equivalent — code path sanity check)
    run_attn ATTN00 \
        --attention_type none

    # ATTN01: SE on layer4, no attention KD
    run_attn ATTN01 \
        --attention_type se \
        --attention_layers layer4 \
        --lambda_att 0.0

    # ATTN02: CBAM on layer4, no attention KD
    run_attn ATTN02 \
        --attention_type cbam \
        --attention_layers layer4 \
        --lambda_att 0.0
fi

# ── ATTN03–ATTN07: core TG-MTA grid (always run) ─────────────────────────────
echo ""
echo "Stage B: TG-MTA Core Grid (ATTN03–ATTN07)"
echo "──────────────────────────────────────────"

# ATTN03: TG-MTA layer4, no attention KD — structure effect only
run_attn ATTN03 \
    --attention_type tgmta \
    --attention_layers layer4 \
    --lambda_att 0.0

# ATTN04: TG-MTA layer3+layer4, no attention KD — deeper layer coverage
run_attn ATTN04 \
    --attention_type tgmta \
    --attention_layers layer3,layer4 \
    --lambda_att 0.0

# ATTN05: TG-MTA layer4, small attention KD (λ=0.01)
run_attn ATTN05 \
    --attention_type tgmta \
    --attention_layers layer4 \
    --lambda_att 0.01 \
    --attn_loss_mode mse_prob

# ATTN06: TG-MTA layer4, medium attention KD (λ=0.05)
run_attn ATTN06 \
    --attention_type tgmta \
    --attention_layers layer4 \
    --lambda_att 0.05 \
    --attn_loss_mode mse_prob

# ATTN07: TG-MTA layer4, large attention KD (λ=0.10)
run_attn ATTN07 \
    --attention_type tgmta \
    --attention_layers layer4 \
    --lambda_att 0.10 \
    --attn_loss_mode mse_prob

# ── optional: ATTN08–ATTN10 (second batch) ──────────────────────────────────
if [ "${RUN_ALL}" = "1" ]; then
    echo ""
    echo "Stage C: Extended Grid (ATTN08–ATTN10)"
    echo "───────────────────────────────────────"

    # ATTN08: TG-MTA layer4, λ=0.05, KL loss (vs MSE in ATTN06)
    run_attn ATTN08 \
        --attention_type tgmta \
        --attention_layers layer4 \
        --lambda_att 0.05 \
        --attn_loss_mode kl

    # ATTN09: TG-MTA layer3+layer4, λ=0.05 (deeper + with att KD)
    run_attn ATTN09 \
        --attention_type tgmta \
        --attention_layers layer3,layer4 \
        --lambda_att 0.05 \
        --attn_loss_mode mse_prob

    # ATTN10: TG-MTA layer4, wide kernel, λ=0.05
    run_attn ATTN10 \
        --attention_type tgmta \
        --attention_layers layer4 \
        --lambda_att 0.05 \
        --attn_loss_mode mse_prob \
        --use_wide_kernel
fi

# ── analysis ──────────────────────────────────────────────────────────────────
echo ""
echo "Summary …"
${PY} -c "
import json, glob, os
out_dir = '${OUT_DIR}'
seed    = ${SEED}
rows = []
for f in sorted(glob.glob(f'{out_dir}/metrics_test_student_II_50hz_f02_ATTN*_seed{seed}.json')):
    d = json.load(open(f))
    vid  = d.get('variant_id', os.path.basename(f))
    auc  = d.get('macro_auc', float('nan'))
    f1   = d.get('macro_f1_tuned', float('nan'))
    att  = d.get('attention_type', '?')
    lays = d.get('attention_layers', '?')
    lam  = d.get('lambda_att', 0)
    rows.append((vid, att, lays, lam, auc, f1))

F02_AUC = 0.8447; F02T1_AUC = 0.8464
print(f'  F02 baseline:  AUC={F02_AUC:.4f}')
print(f'  F02T01 (best): AUC={F02T1_AUC:.4f}')
print()
print(f\"  {'variant':<10} {'att_type':<8} {'layers':<18} {'λ_att':<7} {'AUC':<8} {'F1_t':<7} {'vs F02T01'}\")
print('  ' + '-'*70)
for vid, att, lays, lam, auc, f1 in rows:
    delta = auc - F02T1_AUC
    mark = ' ★' if auc > F02T1_AUC else ''
    print(f'  {vid:<10} {att:<8} {lays:<18} {lam:<7.3f} {auc:<8.4f} {f1:<7.4f} {delta:+.4f}{mark}')
" 2>/dev/null || echo "  (no completed results yet)"

echo ""
echo "============================================================"
echo "  TG-MTA experiments done."
echo "  Results: ${OUT_DIR}/summary.csv"
echo "  Logs:    ${OUT_DIR}/logs/"
echo "============================================================"
