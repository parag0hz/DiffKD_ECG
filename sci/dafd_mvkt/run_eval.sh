#!/bin/bash
# Evaluate all trained models on the test set and print final experiment table.
# Usage: bash dafd_mvkt/run_eval.sh
set -e
cd /home/kwy00/sci

DATA=comper_repo/ptb_xl
OUT=dafd_mvkt/outputs

eval_model() {
    local cfg=$1
    local ckpt=$2
    local name=$3
    echo "--- Evaluating: $name ---"
    conda run -n ecg --no-capture-output python -u dafd_mvkt/evaluate.py \
        --config "$cfg" \
        --data_dir "$DATA" \
        --ckpt "$ckpt" \
        --split test \
        --model_name "$name" \
        --output_dir "$OUT" \
        --tune_threshold
}

# Teacher
eval_model dafd_mvkt/configs/teacher_500hz.yaml \
           "$OUT/teacher_best.pt" \
           teacher

# 100Hz students
eval_model dafd_mvkt/configs/student_100hz.yaml \
           "$OUT/student_100hz_bce.pt" \
           student_100hz_bce

eval_model dafd_mvkt/configs/student_100hz.yaml \
           "$OUT/student_100hz_mkd.pt" \
           student_100hz_mkd

eval_model dafd_mvkt/configs/student_100hz.yaml \
           "$OUT/student_100hz_mkd_crf.pt" \
           student_100hz_mkd_crf

eval_model dafd_mvkt/configs/student_100hz.yaml \
           "$OUT/student_100hz_full.pt" \
           student_100hz_full

# 50Hz students
eval_model dafd_mvkt/configs/student_50hz.yaml \
           "$OUT/student_50hz_bce.pt" \
           student_50hz_bce

eval_model dafd_mvkt/configs/student_50hz.yaml \
           "$OUT/student_50hz_full.pt" \
           student_50hz_full

echo ""
echo "========================================"
echo "         EXPERIMENT TABLE (test)"
echo "========================================"
printf "%-28s  %8s  %8s\n" "Method" "MacroAUC" "MacroF1"
printf "%-28s  %8s  %8s\n" "----------------------------" "--------" "-------"

for name in teacher student_100hz_bce student_100hz_mkd student_100hz_mkd_crf student_100hz_full student_50hz_bce student_50hz_full; do
    MFILE="$OUT/metrics_test_${name}.json"
    if [ -f "$MFILE" ]; then
        AUC=$(python3 -c "import json; d=json.load(open('$MFILE')); print(f\"{d['macro_auc']:.4f}\")")
        F1=$(python3  -c "import json; d=json.load(open('$MFILE')); print(f\"{d['macro_f1']:.4f}\")")
        printf "%-28s  %8s  %8s\n" "$name" "$AUC" "$F1"
    else
        printf "%-28s  %8s  %8s\n" "$name" "N/A" "N/A"
    fi
done
