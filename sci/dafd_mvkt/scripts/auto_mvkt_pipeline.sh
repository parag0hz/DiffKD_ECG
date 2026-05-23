#!/bin/bash
# Autonomous MVKT-beating pipeline.
#
# Goal: Beat MVKT-ECG (AUC>0.843, F1_tuned>0.626) on PTB-XL 5-class
#       superdiagnostic, single-lead 100Hz classification, then characterise
#       the 50Hz sampling-rate trade-off.
#
# Pipeline phases:
#   Phase 1  — Lead I  100Hz (always)
#   Phase 2  — Lead II 100Hz (always, for complete comparison)
#   Phase 3  — HP tuning      (if RUN_TUNING=1 and below/near target)
#   Phase 4  — 50Hz trade-off (if RUN_50HZ=1 and MVKT beaten)
#   Phase 5  — Multi-seed     (if RUN_MULTISEED_WHEN_SUCCESS=1 and beaten)
#   Final    — Paper tables + status report
#
# Entry command:
#   DATA_DIR=comper_repo/ptb_xl SEED=0 \
#   RUN_TUNING=1 RUN_50HZ=1 RUN_MULTISEED_WHEN_SUCCESS=0 \
#   bash dafd_mvkt/scripts/auto_mvkt_pipeline.sh
#
# Sentinel files in outputs/auto/ prevent re-running completed stages.
# Delete them to force a stage to re-run.

set -e
cd "$(dirname "$0")/../.."

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
TEACHER_CKPT=${TEACHER_CKPT:-"dafd_mvkt/outputs/teacher_best.pt"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
AUTO_DIR="${OUT_DIR}/auto"
RUN_TUNING=${RUN_TUNING:-1}
RUN_50HZ=${RUN_50HZ:-1}
RUN_MULTISEED_WHEN_SUCCESS=${RUN_MULTISEED_WHEN_SUCCESS:-0}

_run_stage() {
    DATA_DIR="${DATA_DIR}" TEACHER_CKPT="${TEACHER_CKPT}" \
    SEED="${SEED}" BATCH_SIZE="${BATCH_SIZE}" OUT_DIR="${OUT_DIR}" \
    bash "$1"
}

mkdir -p "${AUTO_DIR}"

_banner() { echo ""; echo "════════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════════"; }

_collect_and_decide() {
    python dafd_mvkt/experiments/auto_collect_key_results.py \
        --results_dirs "${OUT_DIR}" "${OUT_DIR}/summary" "${OUT_DIR}/tuning" \
        --split test
    python dafd_mvkt/experiments/auto_decider.py \
        --results "${AUTO_DIR}/key_results.json" \
        --target_auc 0.843 \
        --target_f1  0.626
    python dafd_mvkt/experiments/auto_make_status_report.py \
        --status  "${AUTO_DIR}/status.json" \
        --results "${AUTO_DIR}/key_results.json" \
        --out      "${AUTO_DIR}/latest_summary.md"
}

_source_env() {
    if [ -f "${AUTO_DIR}/decision.env" ]; then
        # shellcheck source=/dev/null
        source "${AUTO_DIR}/decision.env"
    fi
}

# ── Preflight ─────────────────────────────────────────────────────────────────
_banner "PREFLIGHT CHECK"
if ! python dafd_mvkt/experiments/auto_check_artifacts.py \
        --data_dir "${DATA_DIR}"; then
    echo "PREFLIGHT FAILED. Aborting."
    exit 1
fi

# ── Initial collect/decide (may have prior results) ──────────────────────────
_banner "Initial result collection"
_collect_and_decide 2>/dev/null || true
_source_env

echo ""
echo "Initial status: MVKT_BEATEN=${MVKT_BEATEN:-0}  NEXT_STAGE=${NEXT_STAGE:-leadI_clecg}"
echo "Best 100Hz AUC: ${BEST_AUC_100HZ:-N/A}  F1: ${BEST_F1_100HZ:-N/A}"

# ── Phase 1: Lead I 100Hz ─────────────────────────────────────────────────────
SENTINEL_LEADI="${AUTO_DIR}/stage_leadI_done"
if [ ! -f "${SENTINEL_LEADI}" ]; then
    _banner "PHASE 1: Lead I 100Hz"
    _run_stage dafd_mvkt/scripts/auto_stage_100hz_leadI.sh
    touch "${SENTINEL_LEADI}"
    _banner "Phase 1 complete — collecting results"
    _collect_and_decide
    _source_env
    echo ""
    echo "Post-LeadI: MVKT_BEATEN=${MVKT_BEATEN:-0}  NEXT_STAGE=${NEXT_STAGE}  AUC=${BEST_AUC_100HZ:-N/A}"
else
    echo "Phase 1 (Lead I) already done — skipping."
fi

# ── Phase 2: Lead II 100Hz ────────────────────────────────────────────────────
SENTINEL_LEADII="${AUTO_DIR}/stage_leadII_done"
if [ ! -f "${SENTINEL_LEADII}" ]; then
    _banner "PHASE 2: Lead II 100Hz"
    _run_stage dafd_mvkt/scripts/auto_stage_100hz_leadII.sh
    touch "${SENTINEL_LEADII}"
    _banner "Phase 2 complete — collecting results"
    _collect_and_decide
    _source_env
    echo ""
    echo "Post-LeadII: MVKT_BEATEN=${MVKT_BEATEN:-0}  NEXT_STAGE=${NEXT_STAGE}  AUC=${BEST_AUC_100HZ:-N/A}"
else
    echo "Phase 2 (Lead II) already done — skipping."
fi

# ── Phase 3: HP Tuning (optional) ────────────────────────────────────────────
SENTINEL_TUNING="${AUTO_DIR}/stage_tuning_done"
if [ "${RUN_TUNING}" = "1" ] && [ ! -f "${SENTINEL_TUNING}" ]; then
    # Run tuning if below target or near target
    _source_env
    MVKT_BEATEN=${MVKT_BEATEN:-0}
    PIPELINE_STATUS=${PIPELINE_STATUS:-"below_target"}
    if [ "${MVKT_BEATEN}" = "0" ] || [ "${PIPELINE_STATUS}" = "near_mvkt" ] || [ "${PIPELINE_STATUS}" = "auc_beaten_f1_short" ]; then
        _banner "PHASE 3: HP Tuning"
        _run_stage dafd_mvkt/scripts/auto_stage_tuning.sh
        touch "${SENTINEL_TUNING}"
        _banner "Phase 3 complete — collecting results"
        _collect_and_decide
        _source_env
        echo ""
        echo "Post-Tuning: MVKT_BEATEN=${MVKT_BEATEN:-0}  AUC=${BEST_AUC_100HZ:-N/A}"
    else
        echo "Phase 3 (Tuning) skipped — MVKT already beaten."
        touch "${SENTINEL_TUNING}"
    fi
else
    [ "${RUN_TUNING}" != "1" ] && echo "Phase 3 (Tuning) disabled (RUN_TUNING=0)."
    [ -f "${SENTINEL_TUNING}" ] && echo "Phase 3 (Tuning) already done — skipping."
fi

# ── Phase 4: 50Hz Trade-off (optional) ───────────────────────────────────────
SENTINEL_50HZ="${AUTO_DIR}/stage_50hz_done"
if [ "${RUN_50HZ}" = "1" ] && [ ! -f "${SENTINEL_50HZ}" ]; then
    _source_env
    MVKT_BEATEN=${MVKT_BEATEN:-0}
    if [ "${MVKT_BEATEN}" = "1" ]; then
        _banner "PHASE 4: 50Hz Trade-off"
        _run_stage dafd_mvkt/scripts/auto_stage_50hz_tradeoff.sh
        touch "${SENTINEL_50HZ}"
        _banner "Phase 4 complete — collecting results"
        _collect_and_decide
        _source_env
        echo ""
        echo "Post-50Hz: TRADEOFF_STATUS=${TRADEOFF_STATUS:-N/A}  BEST_50HZ_AUC=${BEST_50HZ_AUC:-N/A}"
    else
        echo "Phase 4 (50Hz) skipped — MVKT not beaten yet."
    fi
else
    [ "${RUN_50HZ}" != "1" ] && echo "Phase 4 (50Hz) disabled (RUN_50HZ=0)."
    [ -f "${SENTINEL_50HZ}" ] && echo "Phase 4 (50Hz) already done — skipping."
fi

# ── Phase 5: Multi-seed (optional) ───────────────────────────────────────────
SENTINEL_MULTI="${AUTO_DIR}/stage_multiseed_done"
if [ "${RUN_MULTISEED_WHEN_SUCCESS}" = "1" ] && [ ! -f "${SENTINEL_MULTI}" ]; then
    _source_env
    MVKT_BEATEN=${MVKT_BEATEN:-0}
    if [ "${MVKT_BEATEN}" = "1" ]; then
        _banner "PHASE 5: Multi-seed Validation"
        _run_stage dafd_mvkt/scripts/auto_stage_multiseed.sh
        touch "${SENTINEL_MULTI}"
        _banner "Phase 5 complete — collecting results"
        _collect_and_decide
        _source_env
    else
        echo "Phase 5 (Multi-seed) skipped — MVKT not beaten."
    fi
else
    [ "${RUN_MULTISEED_WHEN_SUCCESS}" != "1" ] && echo "Phase 5 (Multi-seed) disabled (RUN_MULTISEED_WHEN_SUCCESS=0)."
    [ -f "${SENTINEL_MULTI}" ] && echo "Phase 5 (Multi-seed) already done — skipping."
fi

# ── Final: Paper tables ───────────────────────────────────────────────────────
_banner "Generating paper tables"
mkdir -p "${OUT_DIR}/paper_tables"
python dafd_mvkt/experiments/make_paper_tables.py \
    --results_dirs "${OUT_DIR}" "${OUT_DIR}/tuning" \
    --out_dir "${OUT_DIR}/paper_tables" \
    --split test || echo "WARNING: make_paper_tables.py failed — no results yet."

# ── Final status ──────────────────────────────────────────────────────────────
_source_env
_banner "PIPELINE COMPLETE"
echo ""
cat "${AUTO_DIR}/latest_summary.md" 2>/dev/null || true
echo ""
echo "Outputs:"
echo "  Status report : ${AUTO_DIR}/latest_summary.md"
echo "  Key results   : ${AUTO_DIR}/key_results.json"
echo "  Decision log  : ${AUTO_DIR}/decision_log.md"
echo "  Paper tables  : ${OUT_DIR}/paper_tables/paper_tables.md"
echo ""
if [ "${MVKT_BEATEN:-0}" = "1" ]; then
    echo "✓ MVKT-ECG BEATEN  AUC=${BEST_AUC_100HZ}  F1=${BEST_F1_100HZ}  lead=${BEST_LEAD_100HZ}"
else
    echo "✗ MVKT-ECG NOT YET BEATEN  best AUC=${BEST_AUC_100HZ:-N/A}"
fi
