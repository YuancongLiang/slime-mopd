#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PRESET:?Set MODEL_PRESET to a scripts/models/*.sh file}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the teacher HF config/tokenizer directory}"
: "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to frozen HF or Megatron weights}"
: "${TEACHER_ID:?Set TEACHER_ID to its registry id}"
: "${TEACHER_VERSION:?Set TEACHER_VERSION to its immutable registry version}"
TEACHER_TP=${TEACHER_TP:-1}
TEACHER_PORT=${TEACHER_PORT:-7999}
export CUDA_DEVICE_MAX_CONNECTIONS=1
if [[ -n ${MEGATRON_ROOT:-} ]]; then
  export PYTHONPATH="$MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
source "$MODEL_PRESET"

exec torchrun --standalone --nproc-per-node="$TEACHER_TP" -m slime.opd.teacher_server \
  "${MODEL_ARGS[@]}" \
  --debug-train-only --actor-num-nodes 1 --actor-num-gpus-per-node "$TEACHER_TP" \
  --hf-checkpoint "$HF_CHECKPOINT" --load "$TEACHER_CHECKPOINT" \
  --opd-server-id "$TEACHER_ID" --opd-server-version "$TEACHER_VERSION" \
  --opd-server-port "$TEACHER_PORT" \
  --tensor-model-parallel-size "$TEACHER_TP" --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
  --bf16 --micro-batch-size 1 --global-batch-size 1 --rollout-batch-size 1 --num-rollout 0 \
  --seq-length 32768 --max-position-embeddings 32768 \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --qwen-gdn-backend fla "$@"
