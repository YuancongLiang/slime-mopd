#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the student HF directory}"
: "${STUDENT_CHECKPOINT:?Set STUDENT_CHECKPOINT to the common initial weights}"
: "${PROMPT_DATA:?Set PROMPT_DATA to the source domain dataset}"
: "${REPLAY_DATA:?Set REPLAY_DATA to the saved rollout path template}"
: "${OPD_BENCHMARK_TEACHER_ID:?Set a common immutable teacher identity for both runs}"
MODE=${1:?Usage: run-opd-replay.sh sampled|full [training arguments]}
shift
case "$MODE" in
  sampled)
    ENTRY=train.py
    OBJECTIVE_ARGS=(--opd-objective sampled --opd-kl-coef 1)
    if [[ -n ${SAMPLED_DOMAIN_CONFIG:-} ]]; then
      OBJECTIVE_ARGS+=(--opd-type sglang --opd-domain-config "$SAMPLED_DOMAIN_CONFIG")
    else
      : "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT or SAMPLED_DOMAIN_CONFIG}"
      OBJECTIVE_ARGS+=(--opd-type megatron --opd-teacher-load "$TEACHER_CHECKPOINT")
    fi
    ;;
  full)
    : "${OPD_CONFIG:?Set OPD_CONFIG to the hidden teacher registry}"
    ENTRY=train_mopd.py
    OBJECTIVE_ARGS=(--opd-objective full_vocab_reverse_kl --opd-config "$OPD_CONFIG")
    ;;
  *) echo "Expected sampled or full" >&2; exit 2 ;;
esac
export SAVE_DIR=${SAVE_DIR:-${BENCHMARK_DIR:-/tmp/opd-comparison}/$MODE}
export CUDA_DEVICE_MAX_CONNECTIONS=1
if [[ -n ${MEGATRON_ROOT:-} ]]; then
  export PYTHONPATH="$MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
source "${MODEL_PRESET:-scripts/models/qwen3.5-2B.sh}"
source examples/on_policy_distillation/wandb-env.sh "qwen3.5-2B-opd-replay-$MODE"

exec python "$ENTRY" "${MODEL_ARGS[@]}" "${WANDB_ARGS[@]}" "${OBJECTIVE_ARGS[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" --load "$STUDENT_CHECKPOINT" --save "$SAVE_DIR" --finetune \
  --actor-num-nodes 1 --actor-num-gpus-per-node "${LEARNER_GPUS:-4}" --rollout-num-gpus 1 \
  --tensor-model-parallel-size "${LEARNER_TP:-1}" --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
  --prompt-data "$PROMPT_DATA" --input-key prompt --metadata-key metadata --apply-chat-template \
  --rollout-batch-size "${GLOBAL_BATCH:-64}" --global-batch-size "${GLOBAL_BATCH:-64}" \
  --n-samples-per-prompt 1 --num-rollout "${NUM_ROLLOUT:-12}" --load-debug-rollout-data "$REPLAY_DATA" \
  --bf16 --micro-batch-size 1 --use-dynamic-batch-size --max-tokens-per-gpu "${PACKED_TOKENS:-4096}" \
  --seq-length 8192 --max-position-embeddings 8192 --rollout-temperature 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --optimizer adam --lr "${LEARNING_RATE:-1e-6}" --lr-decay-style constant \
  --kl-coef 0 --entropy-coef 0 --attention-dropout 0 --hidden-dropout 0 \
  --accumulate-allreduce-grads-in-fp32 --attention-backend flash --qwen-gdn-backend fla \
  --use-opd --opd-refresh-replay-targets --opd-metrics-warmup 2 --opd-metrics-interval "${OPD_METRICS_INTERVAL:-1}" \
  --opd-benchmark-id "${OPD_BENCHMARK_ID:-qwen35-2b-fixed-v1}" \
  --opd-benchmark-teacher-id "$OPD_BENCHMARK_TEACHER_ID" "$@"
