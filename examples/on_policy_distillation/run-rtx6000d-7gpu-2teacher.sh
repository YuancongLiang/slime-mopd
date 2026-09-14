#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

# The dedicated Ray head must also start with this five-GPU visibility mask.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:16379}
export MODEL_PRESET=scripts/models/qwen3.5-2B.sh
export OPD_CONFIG=${OPD_CONFIG:-examples/on_policy_distillation/rtx6000d-7gpu-2teacher.yaml}
export LEARNER_NODES=1 LEARNER_GPUS_PER_NODE=4 LEARNER_TP=1 LEARNER_EP=1
export ROLLOUT_GPUS=1 ROLLOUT_TP=1
export GLOBAL_BATCH=${GLOBAL_BATCH:-64}
export PACKED_TOKENS=${PACKED_TOKENS:-4096}
export NUM_ROLLOUT=${NUM_ROLLOUT:-100}
export SAVE_INTERVAL=${SAVE_INTERVAL:-10}

if (( GLOBAL_BATCH <= 0 || GLOBAL_BATCH % 4 != 0 )); then
  echo "GLOBAL_BATCH must be positive and divisible by learner DP=4" >&2
  exit 2
fi

exec bash examples/on_policy_distillation/run-full-vocab.sh \
  --num-gpus-per-node 5 \
  --rollout-max-context-len 8192 --rollout-max-response-len 4096 \
  --seq-length 8192 --max-position-embeddings 8192 \
  "$@"
