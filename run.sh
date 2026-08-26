#!/usr/bin/env bash
# Reproduce Stage-2 on the shipped Cora / CiteSeer embeddings.
# Override GPU / datasets without editing this file:
#   DEVICE=0 DATASETS="cora" NUM_SEEDS=1 bash run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs

DEVICE="${DEVICE:-auto}"
DATASETS="${DATASETS:-citeseer cora}"
NUM_SEEDS="${NUM_SEEDS:-3}"
MODEL_TYPE="${MODEL_TYPE:-hitec_ngs}"
SUB_RATE="${SUB_RATE:-0.3}"
S_WALK="${S_WALK:-3}"

export PYTHONUNBUFFERED=1

echo "[run] device=${DEVICE}  datasets=${DATASETS}  seeds=${NUM_SEEDS}"
python3 scripts/check_setup.py

for dname in ${DATASETS}; do
    log="logs/${dname}_${MODEL_TYPE}_s${S_WALK}.out"
    echo "[run] ${dname} -> ${log}"
    python3 train.py \
        --dataset "${dname}" \
        --device "${DEVICE}" \
        --model_type "${MODEL_TYPE}" \
        --s_walk "${S_WALK}" \
        --num_seeds "${NUM_SEEDS}" \
        --sub_rate "${SUB_RATE}" \
        | tee -a "${log}"
done

echo "[ok] finished. Logs under ${ROOT}/logs/"
