#!/bin/bash
# Pretraining comparison: Random init vs CLECG vs SimCLR for HST-KD.
#
# Runs all SimCLR downstream experiments and produces a comparison report.
# Assumes CLECG results already exist from the main auto pipeline.
#
# Phases (sequential, max VRAM per run):
#   1. Lead II 100Hz: SimCLR student-only HST
#   2. Lead II 100Hz: SimCLR TA+Student HST (if RUN_TA_SIMCLR=1)
#   3. Lead I 100Hz:  SimCLR student-only HST
#   4. Lead II 50Hz:  SimCLR progressive student (if RUN_50HZ=1)
#   5. Aggregate results + print comparison summary
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl SEED=0 RUN_50HZ=1 RUN_TA_SIMCLR=1 \
#   bash dafd_mvkt/scripts/run_pretraining_comparison.sh
#
# Environment variables:
#   DATA_DIR         PTB-XL root
#   SEED             random seed (default: 0)
#   LEADS            leads to run (default: "II I")
#   RUN_50HZ         run 50Hz comparison (default: 1)
#   RUN_TA_SIMCLR    train SimCLR-init TA (default: 1)
#   BATCH_SIZE       batch size (default: 256)
#   OUT_DIR          output dir (default: dafd_mvkt/outputs)

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
SEED=${SEED:-0}
LEADS=${LEADS:-"II I"}
RUN_50HZ=${RUN_50HZ:-1}
RUN_TA_SIMCLR=${RUN_TA_SIMCLR:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
AUTO_DIR="${OUT_DIR}/auto"

_banner() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  $*"
    echo "════════════════════════════════════════════════════════"
}

_stage_env() {
    DATA_DIR="${DATA_DIR}" TEACHER_CKPT="${TEACHER_CKPT}" \
    SEED="${SEED}" BATCH_SIZE="${BATCH_SIZE}" OUT_DIR="${OUT_DIR}" \
    RUN_TA_SIMCLR="${RUN_TA_SIMCLR}" "$@"
}

mkdir -p "${AUTO_DIR}"

_banner "PRETRAINING COMPARISON: CLECG vs SimCLR"
echo "DATA_DIR=${DATA_DIR}  SEED=${SEED}  LEADS=${LEADS}"
echo "RUN_50HZ=${RUN_50HZ}  RUN_TA_SIMCLR=${RUN_TA_SIMCLR}"

# ── 1. Inspect comper_repo SimCLR ────────────────────────────────────────────
_banner "Step 1: comper_repo SimCLR inspection"
echo "Architecture mismatch already confirmed: ConvNeXt (comper_repo) vs ResNet1d (dafd_mvkt)."
echo "Using dafd_mvkt-native SimCLR pretraining (pretrain_simclr.py)."
echo "See: ${AUTO_DIR}/simclr_repo_inspection.md"

# ── 2. Lead II 100Hz SimCLR experiments ──────────────────────────────────────
_banner "Step 2: Lead II 100Hz — SimCLR vs CLECG vs Random"
_stage_env bash dafd_mvkt/scripts/run_simclr_hst_100hz_leadII.sh

# ── 3. Lead I 100Hz SimCLR experiments ───────────────────────────────────────
_banner "Step 3: Lead I 100Hz — SimCLR vs CLECG vs Random"
_stage_env bash dafd_mvkt/scripts/run_simclr_hst_100hz_leadI.sh

# ── 4. Lead II 50Hz SimCLR experiments ───────────────────────────────────────
if [ "${RUN_50HZ}" = "1" ]; then
    _banner "Step 4: Lead II 50Hz — SimCLR Progressive HST-KD"
    _stage_env bash dafd_mvkt/scripts/run_simclr_progressive_50hz_leadII.sh
else
    echo "Step 4 (50Hz) skipped (RUN_50HZ=0)."
fi

# ── 5. Aggregate results ──────────────────────────────────────────────────────
_banner "Step 5: Aggregate results"
mkdir -p "${OUT_DIR}/summary"
python dafd_mvkt/experiments/aggregate_results.py \
    --results_dir "${OUT_DIR}" \
    --out_dir "${OUT_DIR}/summary" \
    --split test

# ── 6. Build pretraining comparison report ────────────────────────────────────
_banner "Step 6: Pretraining comparison report"
python dafd_mvkt/experiments/make_pretraining_report.py \
    --results_dirs "${OUT_DIR}" "${OUT_DIR}/tuning" \
    --out "${AUTO_DIR}/pretraining_comparison_report.md" \
    --split test

# ── 7. Print comparison summary ──────────────────────────────────────────────
_banner "COMPARISON SUMMARY"
echo ""
cat "${AUTO_DIR}/pretraining_comparison_report.md" 2>/dev/null || echo "(report not yet generated)"
echo ""
echo "Full results: ${OUT_DIR}/summary/summary_results.md"
echo "Report:       ${AUTO_DIR}/pretraining_comparison_report.md"
