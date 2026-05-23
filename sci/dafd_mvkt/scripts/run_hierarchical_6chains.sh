#!/usr/bin/env bash
# Run all 6 hierarchical chain KD experiments.
#
# Usage:
#   DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II REUSE_INTERMEDIATE=0 \
#   bash dafd_mvkt/scripts/run_hierarchical_6chains.sh

set -euo pipefail

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}
REUSE_INTERMEDIATE=${REUSE_INTERMEDIATE:-0}

CONFIG="dafd_mvkt/configs/hierarchical_chains_6.yaml"
OUT_DIR="dafd_mvkt/outputs/hierchain"
SIMCLR_CKPT="dafd_mvkt/outputs/simclr/simclr_II_50hz_seed${SEED}_encoder.pt"

echo "=============================================="
echo "  Hierarchical 6-Chain KD Screening"
echo "  DATA_DIR=${DATA_DIR}  LEAD=${LEAD}  SEED=${SEED}"
echo "  REUSE_INTERMEDIATE=${REUSE_INTERMEDIATE}"
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

# ── Step 3: run 6 chains ──────────────────────────────────────────────────
CHAINS=(H01 H02 H03 H04 H05 H06)

echo ""
echo "Step 3: Running 6 hierarchical chains …"

REUSE_FLAG=""
if [ "${REUSE_INTERMEDIATE}" = "1" ]; then
    REUSE_FLAG="--reuse_intermediate"
fi

for CHAIN in "${CHAINS[@]}"; do
    echo ""
    echo "── ${CHAIN} ──────────────────────────────"

    # determine final student metrics path
    # H01: 12L500_1L500_1L100  H02: 12L100_1L500_1L100  H03: 1L500_1L100_1L50
    # H04: 12L100_1L100_1L50   H05: 12L100_12L50_1L50   H06: 12L500_12L100_1L100
    case "${CHAIN}" in
        H01) VIEWS="12L500_1L500_1L100" ;;
        H02) VIEWS="12L100_1L500_1L100" ;;
        H03) VIEWS="1L500_1L100_1L50" ;;
        H04) VIEWS="12L100_1L100_1L50" ;;
        H05) VIEWS="12L100_12L50_1L50" ;;
        H06) VIEWS="12L500_12L100_1L100" ;;
    esac

    FINAL_METRICS="${OUT_DIR}/metrics_test_student_II_50hz_hier_${CHAIN}_${VIEWS}_simclr_seed${SEED}.json"

    if [ -f "${FINAL_METRICS}" ]; then
        echo "   [skip] final metrics already exist: ${FINAL_METRICS}"
        continue
    fi

    python dafd_mvkt/train_hierarchical_chain.py \
        --config   "${CONFIG}" \
        --data_dir "${DATA_DIR}" \
        --lead     "${LEAD}" \
        --chain_id "${CHAIN}" \
        --seed     "${SEED}" \
        ${REUSE_FLAG}

done

# ── Step 4: aggregate ─────────────────────────────────────────────────────
echo ""
echo "Step 4: Aggregating results …"
python dafd_mvkt/experiments/analyze_hierarchical_chains.py \
    --seed "${SEED}" \
    --output_dir "${OUT_DIR}"

echo ""
echo "=============================================="
echo "  All 6 chains complete."
echo "=============================================="
echo ""
echo "Results:"
cat "${OUT_DIR}/hierarchical_6chains_results.md" 2>/dev/null || true
echo ""
cat "${OUT_DIR}/hierarchical_chain_analysis.md" 2>/dev/null || true
