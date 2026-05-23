#!/bin/bash
# CLECG MoCo contrastive pretraining.
#
# Trains CLECG encoders for:
#   - Lead II @ 500Hz (for TA initialization)
#   - Lead II @ 100Hz (for student initialization)
#   - Lead I  @ 500Hz (for TA initialization)
#   - Lead I  @ 100Hz (for student initialization)
#
# Usage (from ~/sci):
#   DATA_DIR=comper_repo/ptb_xl bash dafd_mvkt/scripts/run_clecg_pretrain.sh
#
# Environment variables:
#   DATA_DIR     path to PTB-XL root
#   SEED         random seed (default: 0)
#   BATCH_SIZE   batch size (default: 256)
#   EPOCHS       training epochs (default: 100)
#   OUT_DIR      output directory (default: dafd_mvkt/outputs)
#   LEADS        comma-separated leads to pretrain (default: "II,I")
#   HZS          comma-separated Hz values (default: "500,100")

set -e
cd "$(dirname "$0")/../.."

DATA_DIR=${DATA_DIR:-"comper_repo/ptb_xl"}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
EPOCHS=${EPOCHS:-100}
OUT_DIR=${OUT_DIR:-"dafd_mvkt/outputs"}
LEADS=${LEADS:-"II,I"}
HZS=${HZS:-"500,100"}

echo "========================================================"
echo "CLECG Pretraining  seed=${SEED}  epochs=${EPOCHS}"
echo "DATA_DIR=${DATA_DIR}  BATCH_SIZE=${BATCH_SIZE}"
echo "Leads: ${LEADS}   Hz: ${HZS}"
echo "========================================================"

run_pretrain() {
    local LEAD=$1
    local HZ=$2
    local RUN_NAME="clecg_${LEAD}_${HZ}hz_seed${SEED}"
    local ENCODER_CKPT="${OUT_DIR}/${RUN_NAME}_encoder.pt"

    echo ""
    echo "════════════════════════════════════════"
    echo "  CLECG pretrain: lead=${LEAD}  hz=${HZ}"
    echo "════════════════════════════════════════"

    if [ -f "${ENCODER_CKPT}" ]; then
        echo "  Encoder exists — skipping."
        return
    fi

    # Select config
    if [ "${LEAD}" = "II" ] && [ "${HZ}" = "500" ]; then
        CONFIG="dafd_mvkt/configs/clecg_leadII_500hz.yaml"
    elif [ "${LEAD}" = "II" ] && [ "${HZ}" = "100" ]; then
        CONFIG="dafd_mvkt/configs/clecg_leadII_100hz.yaml"
    elif [ "${LEAD}" = "I" ] && [ "${HZ}" = "500" ]; then
        CONFIG="dafd_mvkt/configs/clecg_leadI_500hz.yaml"
    elif [ "${LEAD}" = "I" ] && [ "${HZ}" = "100" ]; then
        CONFIG="dafd_mvkt/configs/clecg_leadI_100hz.yaml"
    else
        echo "  No config for lead=${LEAD} hz=${HZ} — skipping."
        return
    fi

    python dafd_mvkt/pretrain_clecg.py \
        --config "${CONFIG}" \
        --data_dir "${DATA_DIR}" \
        --lead "${LEAD}" \
        --hz "${HZ}" \
        --seed "${SEED}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --run_name "${RUN_NAME}" \
        --output_dir "${OUT_DIR}"
}

# Run sequentially
IFS=',' read -ra LEAD_ARR <<< "$LEADS"
IFS=',' read -ra HZ_ARR   <<< "$HZS"

for LEAD in "${LEAD_ARR[@]}"; do
    for HZ in "${HZ_ARR[@]}"; do
        run_pretrain "${LEAD}" "${HZ}"
    done
done

echo ""
echo "CLECG pretraining complete."
echo "Encoder checkpoints in ${OUT_DIR}/clecg_*_encoder.pt"
