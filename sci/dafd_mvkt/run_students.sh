#!/bin/bash
# Run all student experiments sequentially after 100Hz BCE finishes.
# Usage: bash dafd_mvkt/run_students.sh

set -e
cd /home/kwy00/sci

DATA=comper_repo/ptb_xl
TEACHER_CKPT=dafd_mvkt/outputs/teacher_best.pt
TEACHER_CFG=dafd_mvkt/configs/teacher_500hz.yaml

run_student() {
    local cfg=$1
    local losses=$2
    local out_dir=$3
    local ckpt_out=$4
    local log=$5

    echo "=== Starting: losses=$losses  out=$out_dir ==="
    conda run -n ecg --no-capture-output python -u dafd_mvkt/train_student.py \
        --config "$cfg" \
        --teacher_config "$TEACHER_CFG" \
        --data_dir "$DATA" \
        --teacher_ckpt "$TEACHER_CKPT" \
        --losses "$losses" \
        --output_dir "$out_dir" \
        --ckpt_out "$ckpt_out" \
        > "$log" 2>&1
    echo "=== Done: $out_dir ==="
}

# 100Hz MKD only
run_student \
    dafd_mvkt/configs/student_100hz.yaml \
    bce,mkd \
    dafd_mvkt/outputs/student_100hz_mkd \
    dafd_mvkt/outputs/student_100hz_mkd.pt \
    dafd_mvkt/outputs/student_100hz_mkd.log

# 100Hz MKD+CRF
run_student \
    dafd_mvkt/configs/student_100hz.yaml \
    bce,mkd,crf \
    dafd_mvkt/outputs/student_100hz_mkd_crf \
    dafd_mvkt/outputs/student_100hz_mkd_crf.pt \
    dafd_mvkt/outputs/student_100hz_mkd_crf.log

# 100Hz full (MKD+CRF+DAF)
run_student \
    dafd_mvkt/configs/student_100hz.yaml \
    bce,mkd,crf,daf \
    dafd_mvkt/outputs/student_100hz_full \
    dafd_mvkt/outputs/student_100hz_full.pt \
    dafd_mvkt/outputs/student_100hz_full.log

# 50Hz baseline
run_student \
    dafd_mvkt/configs/student_50hz.yaml \
    bce \
    dafd_mvkt/outputs/student_50hz_bce \
    dafd_mvkt/outputs/student_50hz_bce.pt \
    dafd_mvkt/outputs/student_50hz_bce.log

# 50Hz full (MKD+CRF+DAF)
run_student \
    dafd_mvkt/configs/student_50hz.yaml \
    bce,mkd,crf,daf \
    dafd_mvkt/outputs/student_50hz_full \
    dafd_mvkt/outputs/student_50hz_full.pt \
    dafd_mvkt/outputs/student_50hz_full.log

echo "All student experiments complete!"
