#!/usr/bin/env bash
# Run all 6 fork-join dual distillation experiments.
#
# Usage:
#   DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II REUSE_BRANCHES=0 \
#   bash dafd_mvkt/scripts/run_fork_join_6.sh

set -euo pipefail

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}
REUSE_BRANCHES=${REUSE_BRANCHES:-0}

CONFIG="dafd_mvkt/configs/fork_join_dual_6.yaml"
OUT_DIR="dafd_mvkt/outputs/forkjoin"
SIMCLR_CKPT="dafd_mvkt/outputs/simclr/simclr_II_50hz_seed${SEED}_encoder.pt"

echo "=============================================="
echo "  Fork-Join Dual Distillation Screening"
echo "  DATA_DIR=${DATA_DIR}  LEAD=${LEAD}  SEED=${SEED}"
echo "  REUSE_BRANCHES=${REUSE_BRANCHES}"
echo "=============================================="

mkdir -p "${OUT_DIR}"

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

# ── Step 2: SimCLR init ───────────────────────────────────────────────────
echo ""
echo "Step 2: Checking SimCLR student init …"
if [ ! -f "${SIMCLR_CKPT}" ]; then
    echo "  [WARN] SimCLR encoder not found: ${SIMCLR_CKPT}"
    echo "  Running SimCLR pretraining for 1-lead II 50Hz …"
    python dafd_mvkt/pretrain_simclr.py \
        --data_dir "${DATA_DIR}" \
        --lead     "${LEAD}" \
        --sampling_rate 50 \
        --seed     "${SEED}" \
        --output_dir "dafd_mvkt/outputs/simclr"
fi
echo "  SimCLR init: ${SIMCLR_CKPT}"

# ── Step 3: run 6 fork-join experiments ──────────────────────────────────
FORKS=(F01 F02 F03 F04 F05 F06)

echo ""
echo "Step 3: Running 6 fork-join experiments …"

REUSE_FLAG=""
if [ "${REUSE_BRANCHES}" = "1" ]; then
    REUSE_FLAG="--reuse_branches"
fi

for FORK in "${FORKS[@]}"; do
    echo ""
    echo "── ${FORK} ──────────────────────────────"

    # determine final student metrics path
    case "${FORK}" in
        F01) T1="12L100"; T2="12L50";  T3="1L100" ;;
        F02) T1="12L100"; T2="1L500";  T3="1L100" ;;
        F03) T1="12L500"; T2="12L50";  T3="1L100" ;;
        F04) T1="12L500"; T2="1L500";  T3="1L100" ;;
        F05) T1="12L100"; T2="12L50";  T3="1L50"  ;;
        F06) T1="1L500";  T2="1L100";  T3="1L50"  ;;
    esac

    FINAL_METRICS="${OUT_DIR}/metrics_test_student_II_50hz_forkjoin_${FORK}_${T1}_to_${T2}_${T3}_simclr_seed${SEED}.json"

    if [ -f "${FINAL_METRICS}" ]; then
        echo "   [skip] final metrics already exist: ${FINAL_METRICS}"
        continue
    fi

    python dafd_mvkt/train_fork_join_dual.py \
        --config   "${CONFIG}" \
        --data_dir "${DATA_DIR}" \
        --lead     "${LEAD}" \
        --fork_id  "${FORK}" \
        --seed     "${SEED}" \
        ${REUSE_FLAG}

done

# ── Step 4: aggregate ─────────────────────────────────────────────────────
echo ""
echo "Step 4: Aggregating results …"
python dafd_mvkt/experiments/analyze_fork_join.py \
    --seed "${SEED}" \
    --output_dir "${OUT_DIR}"

echo ""
echo "=============================================="
echo "  All 6 fork-join experiments complete."
echo "=============================================="
echo ""
echo "Results:"
cat "${OUT_DIR}/fork_join_6_results.md" 2>/dev/null || true
echo ""
cat "${OUT_DIR}/fork_join_analysis.md" 2>/dev/null || true
