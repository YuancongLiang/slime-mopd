#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PRESET:?Set MODEL_PRESET to a scripts/models/*.sh file}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the student HF config/tokenizer directory}"
: "${STUDENT_CHECKPOINT:?Set STUDENT_CHECKPOINT to initial or resumed weights}"
: "${OPD_CONFIG:?Set OPD_CONFIG to the full-vocabulary YAML}"
: "${PROMPT_DATA:?Set PROMPT_DATA to domain-annotated prompt data}"
: "${SAVE_DIR:?Set SAVE_DIR to the output checkpoint directory}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
if [[ -n ${MEGATRON_ROOT:-} ]]; then
  export PYTHONPATH="$MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
source "$MODEL_PRESET"

exec python train_mopd.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" --load "$STUDENT_CHECKPOINT" --save "$SAVE_DIR" \
  --save-interval "${SAVE_INTERVAL:-20}" \
  --use-opd --opd-objective full_vocab_reverse_kl --opd-config "$OPD_CONFIG" \
  --actor-num-nodes "${LEARNER_NODES:-1}" --actor-num-gpus-per-node "${LEARNER_GPUS_PER_NODE:-8}" \
  --rollout-num-gpus "${ROLLOUT_GPUS:-8}" --rollout-num-gpus-per-engine "${ROLLOUT_TP:-1}" \
  --tensor-model-parallel-size "${LEARNER_TP:-1}" --expert-model-parallel-size "${LEARNER_EP:-1}" \
  --expert-tensor-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --prompt-data "$PROMPT_DATA" --input-key prompt --metadata-key metadata \
  --rollout-shuffle --apply-chat-template \
  --rollout-batch-size "${GLOBAL_BATCH:-128}" --global-batch-size "${GLOBAL_BATCH:-128}" \
  --n-samples-per-prompt 1 --num-rollout "${NUM_ROLLOUT:-300}" \
  --rollout-max-context-len 32768 --rollout-max-response-len 16384 \
  --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 \
  --use-dynamic-batch-size --max-tokens-per-gpu "${PACKED_TOKENS:-8192}" \
  --bf16 --micro-batch-size 1 --seq-length 32768 --max-position-embeddings 32768 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --optimizer adam --lr "${LEARNING_RATE:-1e-6}" --lr-decay-style constant \
  --kl-coef 0 --entropy-coef 0 --attention-dropout 0 --hidden-dropout 0 \
  --accumulate-allreduce-grads-in-fp32 --attention-backend flash --qwen-gdn-backend fla \
  "$@"
