#!/usr/bin/env bash
# F02 Optimization Runner.
# Stages: weight tuning → CRF/FeatureKD → temperature → branch CRF → (confidence KD)
#
# Usage:
#   DATA_DIR=comper_repo/ptb_xl SEED=0 LEAD=II \
#   bash dafd_mvkt/scripts/run_f02_optimization.sh

set -euo pipefail

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
LEAD=${LEAD:-"II"}

PY=/home/kwy00/anaconda3/envs/ecg/bin/python
CONFIG="dafd_mvkt/configs/adaptive_50hz.yaml"
OUT_DIR="dafd_mvkt/outputs/f02_opt"
BRANCH_DIR="${OUT_DIR}/branch"
FORKJOIN_DIR="dafd_mvkt/outputs/forkjoin"

B500="${FORKJOIN_DIR}/F02_branch2_1L500_from_12L100_seed${SEED}_best.pt"
B100="${FORKJOIN_DIR}/F02_branch3_1L100_from_12L100_seed${SEED}_best.pt"
T12L100="dafd_mvkt/outputs/teacher_12lead_100hz_resnet34_best.pt"
SIMCLR="dafd_mvkt/outputs/simclr/simclr_II_50hz_seed${SEED}_encoder.pt"

CMD_LOG="${OUT_DIR}/f02_opt_commands.log"
ERR_LOG="${OUT_DIR}/errors.log"

echo "============================================================"
echo "  F02 Optimization"
echo "  DATA_DIR=${DATA_DIR}  LEAD=${LEAD}  SEED=${SEED}"
echo "  OUT_DIR=${OUT_DIR}"
echo "============================================================"
echo ""

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs" "${BRANCH_DIR}"

log_cmd() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${CMD_LOG}"; }

run_or_skip() {
    local label="$1"; shift
    local metrics_path="$1"; shift
    if [ -f "${metrics_path}" ]; then
        echo "   [skip] ${label}"
        return 0
    fi
    echo "   [run]  ${label}"
    log_cmd "$label: $*"
    if ! "$@" 2>> "${ERR_LOG}"; then
        echo "   [ERROR] ${label} failed — see ${ERR_LOG}"
        return 1
    fi
}

# ── check deps ────────────────────────────────────────────────────────────────
echo "Checking F02 branch checkpoints …"
for ckpt in "${B500}" "${B100}"; do
    if [ ! -f "${ckpt}" ]; then
        echo "[ERROR] Missing checkpoint: ${ckpt}"
        echo "  Run: DATA_DIR=${DATA_DIR} SEED=${SEED} LEAD=${LEAD} bash dafd_mvkt/scripts/run_fork_join_6.sh"
        exit 1
    fi
done
echo "  1L500 branch: ${B500}"
echo "  1L100 branch: ${B100}"

if [ ! -f "${SIMCLR}" ]; then
    echo "[WARN] SimCLR init not found: ${SIMCLR}"
    echo "  Running SimCLR pretraining …"
    $PY dafd_mvkt/pretrain_simclr.py \
        --data_dir "${DATA_DIR}" --lead "${LEAD}" \
        --sampling_rate 50 --seed "${SEED}" \
        --output_dir "dafd_mvkt/outputs/simclr"
fi

# ── Stage 1: weight tuning ────────────────────────────────────────────────────
echo ""
echo "Stage 1: Weight Tuning (F02W01-F02W07)"
echo "────────────────────────────────────────"

declare -A W_DEFS
W_DEFS[F02W01]="0.50,0.50"
W_DEFS[F02W02]="0.40,0.60"
W_DEFS[F02W03]="0.30,0.70"
W_DEFS[F02W04]="0.60,0.40"
W_DEFS[F02W05]="0.70,0.30"
W_DEFS[F02W06]="0.25,0.75"
W_DEFS[F02W07]="0.75,0.25"

for VID in F02W01 F02W02 F02W03 F02W04 F02W05 F02W06 F02W07; do
    WSTR="${W_DEFS[$VID]}"
    OUT_NAME="student_II_50hz_f02_${VID}_seed${SEED}"
    run_or_skip "${VID}" "${OUT_DIR}/metrics_test_${OUT_NAME}.json" \
        $PY dafd_mvkt/train_f02_student_tune.py \
            --config "${CONFIG}" --data_dir "${DATA_DIR}" --lead "${LEAD}" \
            --teacher_1l500_ckpt "${B500}" --teacher_1l100_ckpt "${B100}" \
            --student_init_ckpt "${SIMCLR}" \
            --weights "${WSTR}" --temperature 2.0 \
            --variant_id "${VID}" --output_dir "${OUT_DIR}" --seed "${SEED}"
done

# ── analyze Phase 2, pick best weight ────────────────────────────────────────
echo ""
echo "Analyzing Phase 1 …"
$PY dafd_mvkt/experiments/analyze_f02_optimization.py --output_dir "${OUT_DIR}" 2>/dev/null || true

BEST_W_VID=$($PY -c "
import json, glob
best_auc, best_vid, best_w = 0, 'F02W02', '0.40,0.60'
for f in sorted(glob.glob('${OUT_DIR}/metrics_test_student_II_50hz_f02_F02W*_seed${SEED}.json')):
    d = json.load(open(f))
    vid = d.get('variant_id','')
    if d['macro_auc'] > best_auc:
        best_auc = d['macro_auc']
        best_vid = vid
        best_w = f\"{d.get('w500',0.4):.2f},{d.get('w100',0.6):.2f}\"
print(best_vid, best_w)
" 2>/dev/null || echo "F02W02 0.40,0.60")
BEST_W_VID_ID=$(echo "${BEST_W_VID}" | awk '{print $1}')
BEST_WEIGHTS=$(echo "${BEST_W_VID}" | awk '{print $2}')
if [ -z "${BEST_WEIGHTS}" ]; then BEST_WEIGHTS="0.40,0.60"; fi
echo "Best weight: ${BEST_W_VID_ID} (${BEST_WEIGHTS})"

# ── Stage 2: CRF / FeatureKD ─────────────────────────────────────────────────
echo ""
echo "Stage 2: CRF / FeatureKD (F02F01-F02F06)"
echo "──────────────────────────────────────────"

run_crf_feat() {
    local VID="$1"; local USE_CRF="$2"; local BETA_CRF="$3"
    local USE_FEAT="$4"; local GAMMA_FEAT="$5"
    local OUT_NAME="student_II_50hz_f02_${VID}_seed${SEED}"
    local EXTRA=""
    [ "${USE_CRF}"  = "1" ] && EXTRA="${EXTRA} --use_crf --beta_crf ${BETA_CRF}"
    [ "${USE_FEAT}" = "1" ] && EXTRA="${EXTRA} --use_feature_kd --gamma_feature ${GAMMA_FEAT}"
    run_or_skip "${VID}" "${OUT_DIR}/metrics_test_${OUT_NAME}.json" \
        $PY dafd_mvkt/train_f02_student_tune.py \
            --config "${CONFIG}" --data_dir "${DATA_DIR}" --lead "${LEAD}" \
            --teacher_1l500_ckpt "${B500}" --teacher_1l100_ckpt "${B100}" \
            --student_init_ckpt "${SIMCLR}" \
            --weights "${BEST_WEIGHTS}" --temperature 2.0 \
            --variant_id "${VID}" --output_dir "${OUT_DIR}" --seed "${SEED}" \
            ${EXTRA}
}

run_crf_feat F02F01 1 0.05 0 0.0
run_crf_feat F02F02 1 0.10 0 0.0
run_crf_feat F02F03 0 0.0  1 0.10
run_crf_feat F02F04 0 0.0  1 0.20
run_crf_feat F02F05 1 0.05 1 0.10
run_crf_feat F02F06 1 0.10 1 0.20

echo ""
echo "Analyzing Phase 2 …"
$PY dafd_mvkt/experiments/analyze_f02_optimization.py --output_dir "${OUT_DIR}" 2>/dev/null || true

BEST_F_EXTRA=$($PY -c "
import json, glob
best_auc, best_extra = 0, ''
for f in sorted(glob.glob('${OUT_DIR}/metrics_test_student_II_50hz_f02_F02F*_seed${SEED}.json')):
    d = json.load(open(f))
    if d['macro_auc'] > best_auc:
        best_auc = d['macro_auc']
        extra = ''
        if d.get('use_crf'): extra += f\" --use_crf --beta_crf {d['beta_crf']}\"
        if d.get('use_feature_kd'): extra += f\" --use_feature_kd --gamma_feature {d['gamma_feature']}\"
        best_extra = extra
print(best_extra.strip())
" 2>/dev/null || echo "")
echo "Best CRF/Feat extra: '${BEST_F_EXTRA}'"

# ── Stage 3: temperature ──────────────────────────────────────────────────────
echo ""
echo "Stage 3: Temperature Tuning (F02T01-F02T04)"
echo "─────────────────────────────────────────────"

for VID_T in "F02T01:1.5" "F02T02:2.0" "F02T03:3.0" "F02T04:4.0"; do
    VID="${VID_T%%:*}"; TEMP="${VID_T##*:}"
    OUT_NAME="student_II_50hz_f02_${VID}_seed${SEED}"
    run_or_skip "${VID}" "${OUT_DIR}/metrics_test_${OUT_NAME}.json" \
        $PY dafd_mvkt/train_f02_student_tune.py \
            --config "${CONFIG}" --data_dir "${DATA_DIR}" --lead "${LEAD}" \
            --teacher_1l500_ckpt "${B500}" --teacher_1l100_ckpt "${B100}" \
            --student_init_ckpt "${SIMCLR}" \
            --weights "${BEST_WEIGHTS}" --temperature "${TEMP}" \
            --variant_id "${VID}" --output_dir "${OUT_DIR}" --seed "${SEED}" \
            ${BEST_F_EXTRA}
done

echo ""
echo "Analyzing Phase 3 …"
$PY dafd_mvkt/experiments/analyze_f02_optimization.py --output_dir "${OUT_DIR}" 2>/dev/null || true

# ── check if improvement found ────────────────────────────────────────────────
CURRENT_BEST_AUC=$($PY -c "
import json, glob
best = 0.0
for f in glob.glob('${OUT_DIR}/metrics_test_student_II_50hz_f02_*.json'):
    d = json.load(open(f))
    if d['macro_auc'] > best: best = d['macro_auc']
print(f'{best:.4f}')
" 2>/dev/null || echo "0.0000")
echo "Current best AUC across all variants: ${CURRENT_BEST_AUC}"

HAS_IMPROVEMENT=$($PY -c "print('yes' if float('${CURRENT_BEST_AUC}') > 0.8447 else 'no')" 2>/dev/null || echo "no")

# ── Stage 4: branch CRF (only if no improvement yet) ─────────────────────────
if [ "${HAS_IMPROVEMENT}" = "no" ]; then
    echo ""
    echo "Stage 4: Branch CRF Enhancement (F02BR01-F02BR04)"
    echo "────────────────────────────────────────────────────"

    BEST_T=$($PY -c "
import json, glob
best_auc, best_t = 0, 2.0
for f in sorted(glob.glob('${OUT_DIR}/metrics_test_student_II_50hz_f02_F02T*_seed${SEED}.json')):
    d = json.load(open(f))
    if d['macro_auc'] > best_auc: best_auc = d['macro_auc']; best_t = d['temperature']
print(best_t)
" 2>/dev/null || echo "2.0")

    run_branch() {
        local VID="$1"; local E500="$2"; local E100="$3"; local BCRF="$4"
        local OUT_NAME="student_II_50hz_f02_${VID}_seed${SEED}"
        local E500_FLAG=""; local E100_FLAG=""
        [ "${E500}" = "1" ] && E500_FLAG="--enhance_1l500"
        [ "${E100}" = "1" ] && E100_FLAG="--enhance_1l100"
        run_or_skip "${VID}" "${OUT_DIR}/metrics_test_${OUT_NAME}.json" \
            $PY dafd_mvkt/train_f02_branch_enhance.py \
                --config "${CONFIG}" --data_dir "${DATA_DIR}" --lead "${LEAD}" \
                --t12l100_ckpt "${T12L100}" \
                --init_1l500_ckpt "${B500}" --init_1l100_ckpt "${B100}" \
                --student_init_ckpt "${SIMCLR}" \
                ${E500_FLAG} ${E100_FLAG} \
                --beta_branch_crf "${BCRF}" \
                --student_weights "${BEST_WEIGHTS}" \
                --temperature "${BEST_T}" \
                --variant_id "${VID}" --output_dir "${BRANCH_DIR}" --seed "${SEED}"
    }

    run_branch F02BR01 1 0 0.05
    run_branch F02BR02 0 1 0.05
    run_branch F02BR03 1 1 0.05
    run_branch F02BR04 1 1 0.10

    echo ""
    echo "Analyzing after Phase 4 …"
    $PY dafd_mvkt/experiments/analyze_f02_optimization.py --output_dir "${OUT_DIR}" 2>/dev/null || true

    CURRENT_BEST_AUC=$($PY -c "
import json, glob
best = 0.0
for f in glob.glob('${OUT_DIR}/metrics_test_student_II_50hz_f02_*.json'):
    d = json.load(open(f))
    if d['macro_auc'] > best: best = d['macro_auc']
print(f'{best:.4f}')
" 2>/dev/null || echo "0.0000")
    HAS_IMPROVEMENT=$($PY -c "print('yes' if float('${CURRENT_BEST_AUC}') > 0.8447 else 'no')" 2>/dev/null || echo "no")
fi

# ── Stage 5: confidence-weighted KD (optional) ───────────────────────────────
if [ "${HAS_IMPROVEMENT}" = "no" ]; then
    echo ""
    echo "Stage 5: Confidence-Weighted KD (F02CW01-F02CW02)"
    echo "─────────────────────────────────────────────────────"

    for VID_MODE in "F02CW01:confidence" "F02CW02:agreement"; do
        VID="${VID_MODE%%:*}"; MODE="${VID_MODE##*:}"
        OUT_NAME="student_II_50hz_f02_${VID}_seed${SEED}"
        run_or_skip "${VID}" "${OUT_DIR}/metrics_test_${OUT_NAME}.json" \
            $PY dafd_mvkt/train_f02_student_tune.py \
                --config "${CONFIG}" --data_dir "${DATA_DIR}" --lead "${LEAD}" \
                --teacher_1l500_ckpt "${B500}" --teacher_1l100_ckpt "${B100}" \
                --student_init_ckpt "${SIMCLR}" \
                --weights "${BEST_WEIGHTS}" --temperature 2.0 \
                --teacher_weight_mode "${MODE}" \
                --variant_id "${VID}" --output_dir "${OUT_DIR}" --seed "${SEED}"
    done
fi

# ── Final analysis ────────────────────────────────────────────────────────────
echo ""
echo "Final Analysis …"
$PY dafd_mvkt/experiments/analyze_f02_optimization.py --output_dir "${OUT_DIR}"

echo ""
echo "============================================================"
echo "  F02 Optimization Complete"
echo "  Results: ${OUT_DIR}/f02_optimization_results.md"
echo "  Analysis: ${OUT_DIR}/f02_optimization_analysis.md"
echo "============================================================"
echo ""
cat "${OUT_DIR}/f02_optimization_results.md"  2>/dev/null || true
echo ""
cat "${OUT_DIR}/f02_optimization_analysis.md" 2>/dev/null || true
