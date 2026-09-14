#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

case "${1:-}" in
  math)
    export TEACHER_ID=math_rl TEACHER_VERSION=math-v1 TEACHER_PORT=7999
    export CUDA_VISIBLE_DEVICES=${TEACHER_GPU:-5}
    ;;
  code)
    export TEACHER_ID=code_rl TEACHER_VERSION=code-v1 TEACHER_PORT=8000
    export CUDA_VISIBLE_DEVICES=${TEACHER_GPU:-6}
    ;;
  *)
    echo "Usage: bash $0 {math|code} [teacher server arguments...]" >&2
    exit 2
    ;;
esac
shift
export MODEL_PRESET=scripts/models/qwen3.5-2B.sh
export TEACHER_TP=1

exec bash examples/on_policy_distillation/serve-hidden.sh \
  --opd-server-host 127.0.0.1 \
  --opd-server-max-context 8192 --opd-server-max-requests 128 \
  --opd-server-max-bytes 2147483648 \
  --seq-length 8192 --max-position-embeddings 8192 \
  "$@"
