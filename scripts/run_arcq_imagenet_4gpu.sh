#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/Quest/AAAI"
CODE="${ROOT}/methods/fair_qat_framework"
CONFIG_LIST="${ROOT}/configs/imagenet_arcq_4gpu/config_list_arcq_imagenet_4gpu.txt"
LOG_DIR="${ROOT}/logs/imagenet_arcq_4gpu"

GPU_COUNT="${GPU_COUNT:-4}"
CONFIG_INDEX="${CONFIG_INDEX:-0}"

if [[ ! -f "${CONFIG_LIST}" ]]; then
  echo "Missing config list: ${CONFIG_LIST}" >&2
  exit 2
fi

CONFIG="$(sed -n "$((CONFIG_INDEX + 1))p" "${CONFIG_LIST}")"
if [[ -z "${CONFIG}" || ! -f "${CONFIG}" ]]; then
  echo "Invalid CONFIG_INDEX=${CONFIG_INDEX}: ${CONFIG}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
cd "${CODE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PATH="/root/miniconda3/bin:${PATH}"

echo "[ARCQ_4GPU] host=$(hostname)"
echo "[ARCQ_4GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "[ARCQ_4GPU] nproc_per_node=${GPU_COUNT}"
echo "[ARCQ_4GPU] config=${CONFIG}"

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${GPU_COUNT}" \
  fair_qat/train_imagenet_qat.py \
  --config "${CONFIG}" \
  --device cuda:0
