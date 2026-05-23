#!/usr/bin/env bash
# Run all adaptive teacher-student distillation experiments.
# Stages:
#   1. E3T01-E3T04 — 3-teacher C14 + EMA regularization
#   2. EPROG01-EPROG04 — Progressive 1L100→1L50 + EMA regularization
#   3. MUT01-MUT04 — Mutual learning between 1L100 and 1L50
#   4. SAF01-SAF04 — Student-aware skip-fork branch distillation
#
# Usage:
#   DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#   bash dafd_mvkt/scripts/run_adaptive_teacher_experiments.sh

set -euo pipefail

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}

CONFIG="dafd_mvkt/configs/adaptive_50hz.yaml"
OUT_DIR="dafd_mvkt/outputs/adaptive"
SKIPFORK_DIR="dafd_mvkt/outputs/skipfork"

# Teacher checkpoints
T12L500="dafd_mvkt/outputs/teacher_best.pt"
T12L100="dafd_mvkt/outputs/teacher_12lead_100hz_resnet34_best.pt"
T1L100="dafd_mvkt/outputs/ta_II_100hz_from_ta500_seed0_best.pt"

SIMCLR_50HZ="dafd_mvkt/outputs/simclr/simclr_II_50hz_seed${SEED}_encoder.pt"
TEACHER_BANK="dafd_mvkt/configs/teacher_bank_II.json"

CMD_LOG="${OUT_DIR}/run_adaptive_commands.log"
ERR_LOG="${OUT_DIR}/run_adaptive_errors.log"

echo "============================================================"
echo "  Adaptive Teacher-Student Distillation Experiments"
echo "  DATA_DIR=${DATA_DIR}  LEAD=${LEAD}  SEED=${SEED}"
echo "  OUT_DIR=${OUT_DIR}"
echo "============================================================"
echo ""

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"

activate_env() {
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate ecg 2>/dev/null || true
    fi
}
activate_env

log_cmd() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${CMD_LOG}"
}

run_or_skip() {
    local label="$1"; shift
    local metrics_path="$1"; shift
    if [ -f "${metrics_path}" ]; then
        echo "   [skip] ${label} — metrics exist: ${metrics_path}"
        return 0
    fi
    echo "   [run]  ${label}"
    log_cmd "$*"
    if ! "$@" 2>> "${ERR_LOG}"; then
        echo "   [ERROR] ${label} failed. See ${ERR_LOG}"
    fi
}

# ── Stage 1a: E3T01-E3T04 — 3-teacher EMA ─────────────────────────────────
echo "Stage 1a: E3T01-E3T04 — 3-teacher C14 + EMA"
echo "─────────────────────────────────────────────"

E3T_COMBOS=(
    "E3T01 C14 0.05 0.99"
    "E3T02 C14 0.1  0.99"
    "E3T03 C14 0.05 0.999"
    "E3T04 C14 0.1  0.999"
)

for entry in "${E3T_COMBOS[@]}"; do
    read -r EID COMBO LAM_EMA DECAY <<< "${entry}"
    LAM_STR=${LAM_EMA/./}
    DECAY_STR=${DECAY/./}
    OUT_NAME="student_II_50hz_3T_${COMBO}_ema_lam${LAM_STR}_decay${DECAY_STR}_seed${SEED}"
    METRICS="${OUT_DIR}/metrics_test_${OUT_NAME}.json"

    run_or_skip "${EID}" "${METRICS}" \
        python dafd_mvkt/train_student_3teacher_ema.py \
            --config              "${CONFIG}" \
            --data_dir            "${DATA_DIR}" \
            --lead                "${LEAD}" \
            --teacher_combo       "${COMBO}" \
            --teacher_ckpts_json  "${TEACHER_BANK}" \
            --student_init_ckpt   "${SIMCLR_50HZ}" \
            --lambda_ema          "${LAM_EMA}" \
            --ema_decay           "${DECAY}" \
            --seed                "${SEED}" \
            --output_name         "${OUT_NAME}" \
            --output_dir          "${OUT_DIR}"
done

echo ""

# ── Stage 1b: EPROG01-EPROG04 — Progressive + EMA ─────────────────────────
echo "Stage 1b: EPROG01-EPROG04 — Progressive 1L100→1L50 + EMA"
echo "────────────────────────────────────────────────────────────"

EPROG_COMBOS=(
    "EPROG01 0.05 0.99"
    "EPROG02 0.1  0.99"
    "EPROG03 0.05 0.999"
    "EPROG04 0.1  0.999"
)

for entry in "${EPROG_COMBOS[@]}"; do
    read -r EID LAM_EMA DECAY <<< "${entry}"
    LAM_STR=${LAM_EMA/./}
    DECAY_STR=${DECAY/./}
    OUT_NAME="student_II_50hz_prog_1l100_ema_lam${LAM_STR}_decay${DECAY_STR}_seed${SEED}"
    METRICS="${OUT_DIR}/metrics_test_${OUT_NAME}.json"

    run_or_skip "${EID}" "${METRICS}" \
        python dafd_mvkt/train_student_progressive_ema.py \
            --config             "${CONFIG}" \
            --data_dir           "${DATA_DIR}" \
            --lead               "${LEAD}" \
            --teacher_ckpt       "${T1L100}" \
            --student_init_ckpt  "${SIMCLR_50HZ}" \
            --lambda_ema         "${LAM_EMA}" \
            --ema_decay          "${DECAY}" \
            --seed               "${SEED}" \
            --output_name        "${OUT_NAME}" \
            --output_dir         "${OUT_DIR}"
done

echo ""

# ── Stage 2: MUT01-MUT04 — Mutual learning ────────────────────────────────
echo "Stage 2: MUT01-MUT04 — Mutual learning 1L100 ⟷ 1L50"
echo "───────────────────────────────────────────────────────"

MUT_COMBOS=(
    "MUT01 12L100 ${T12L100} 0.05"
    "MUT02 12L100 ${T12L100} 0.1"
    "MUT03 12L500 ${T12L500} 0.05"
    "MUT04 12L500 ${T12L500} 0.1"
)

for entry in "${MUT_COMBOS[@]}"; do
    read -r EID UP_VIEW UP_CKPT LAM_MUT <<< "${entry}"
    LAM_STR=${LAM_MUT/./}
    UP_SHORT=${UP_VIEW,,}    # lowercase: 12l100, 12l500
    S_OUT_NAME="student_II_50hz_mut_${UP_SHORT}_lam${LAM_STR}_seed${SEED}"
    T_OUT_NAME="teacher_1l100_mut_${UP_SHORT}_lam${LAM_STR}_seed${SEED}"
    S_METRICS="${OUT_DIR}/metrics_test_${S_OUT_NAME}.json"

    run_or_skip "${EID}" "${S_METRICS}" \
        python dafd_mvkt/train_mutual_1l100_1l50.py \
            --config              "${CONFIG}" \
            --data_dir            "${DATA_DIR}" \
            --lead                "${LEAD}" \
            --upstream_view       "${UP_VIEW}" \
            --upstream_ckpt       "${UP_CKPT}" \
            --teacher_init_ckpt   "${T1L100}" \
            --student_init_ckpt   "${SIMCLR_50HZ}" \
            --lambda_mutual       "${LAM_MUT}" \
            --seed                "${SEED}" \
            --student_output_name "${S_OUT_NAME}" \
            --teacher_output_name "${T_OUT_NAME}" \
            --output_dir          "${OUT_DIR}"
done

echo ""

# ── Stage 3: SAF01-SAF04 — Student-aware skip-fork ────────────────────────
echo "Stage 3: SAF01-SAF04 — Student-aware branch distillation"
echo "──────────────────────────────────────────────────────────"

# Require skip-fork branches to exist
check_skipfork_deps() {
    local sf_id="$1"
    case "${sf_id}" in
        SF01)
            echo "${SKIPFORK_DIR}/SF01_branch2_12L50_from_12L100_seed${SEED}_best.pt"
            echo "${SKIPFORK_DIR}/SF01_branch3_1L100_from_12L100_seed${SEED}_best.pt"
            echo "${SKIPFORK_DIR}/student_II_50hz_skipfork_SF01_12L100_to_12L50_1L100_simclr_seed${SEED}_best.pt"
            ;;
        SF02)
            echo "${SKIPFORK_DIR}/SF02_branch2_1L500_from_12L100_seed${SEED}_best.pt"
            echo "${SKIPFORK_DIR}/SF02_branch3_1L100_from_12L100_seed${SEED}_best.pt"
            echo "${SKIPFORK_DIR}/student_II_50hz_skipfork_SF02_12L100_to_1L500_1L100_simclr_seed${SEED}_best.pt"
            ;;
    esac
}

SAF_COMBOS=(
    "SAF01 SF02 0.05"
    "SAF02 SF02 0.1"
    "SAF03 SF01 0.05"
    "SAF04 SF01 0.1"
)

for entry in "${SAF_COMBOS[@]}"; do
    read -r EID SF_ID LAM_SA <<< "${entry}"
    LAM_STR=${LAM_SA/./}

    case "${SF_ID}" in
        SF01)
            T1_V="12L100"; T2_V="12L50"; T3_V="1L100"
            T2_CKPT="${SKIPFORK_DIR}/SF01_branch2_12L50_from_12L100_seed${SEED}_best.pt"
            T3_CKPT="${SKIPFORK_DIR}/SF01_branch3_1L100_from_12L100_seed${SEED}_best.pt"
            S_CKPT="${SKIPFORK_DIR}/student_II_50hz_skipfork_SF01_12L100_to_12L50_1L100_simclr_seed${SEED}_best.pt"
            ;;
        SF02)
            T1_V="12L100"; T2_V="1L500"; T3_V="1L100"
            T2_CKPT="${SKIPFORK_DIR}/SF02_branch2_1L500_from_12L100_seed${SEED}_best.pt"
            T3_CKPT="${SKIPFORK_DIR}/SF02_branch3_1L100_from_12L100_seed${SEED}_best.pt"
            S_CKPT="${SKIPFORK_DIR}/student_II_50hz_skipfork_SF02_12L100_to_1L500_1L100_simclr_seed${SEED}_best.pt"
            ;;
    esac

    T1_V_LOWER=${T1_V,,}; T2_V_LOWER=${T2_V,,}; T3_V_LOWER=${T3_V,,}
    SF_LOWER=${SF_ID,,}
    OUT_BASE="student_II_50hz_saf_${SF_LOWER}_${T1_V_LOWER}_to_${T2_V_LOWER}_${T3_V_LOWER}_sa${LAM_STR}_seed${SEED}"
    S_METRICS="${OUT_DIR}/metrics_test_${OUT_BASE}.json"

    # check deps
    MISSING=0
    for dep in "${T2_CKPT}" "${T3_CKPT}" "${S_CKPT}"; do
        if [ ! -f "${dep}" ]; then
            echo "   [WARN] ${EID}: missing dependency: ${dep}"
            MISSING=1
        fi
    done
    if [ "${MISSING}" = "1" ]; then
        echo "   [skip] ${EID}: run run_skip_fork_6.sh first"
        continue
    fi

    run_or_skip "${EID}" "${S_METRICS}" \
        python dafd_mvkt/train_student_aware_skip_fork.py \
            --config       "${CONFIG}" \
            --data_dir     "${DATA_DIR}" \
            --lead         "${LEAD}" \
            --sf_id        "${SF_ID}" \
            --t1_ckpt      "${T12L100}" \
            --t2_ckpt      "${T2_CKPT}" \
            --t3_ckpt      "${T3_CKPT}" \
            --student_ckpt "${S_CKPT}" \
            --lambda_sa    "${LAM_SA}" \
            --seed         "${SEED}" \
            --output_base  "${OUT_BASE}" \
            --output_dir   "${OUT_DIR}"
done

echo ""

# ── Stage 4: Analysis ──────────────────────────────────────────────────────
echo "Stage 4: Running analysis …"
python dafd_mvkt/experiments/analyze_adaptive_teacher.py \
    --output_dir "${OUT_DIR}" \
    --seed       "${SEED}"

echo ""
echo "============================================================"
echo "  All adaptive experiments complete."
echo "  Results: ${OUT_DIR}/adaptive_results.md"
echo "  Analysis: ${OUT_DIR}/adaptive_analysis.md"
echo "============================================================"
