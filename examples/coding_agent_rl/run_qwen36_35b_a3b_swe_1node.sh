#!/usr/bin/env bash
# Single-node (8 GPU) smoke configuration for the coding-agent RL example.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export ETP_SIZE="${ETP_SIZE:-1}"

export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-8}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-8}"
export ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-1}"
export ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-8}"
export ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.70}"

# Keep the first end-to-end run deliberately small. Increase these after one
# rollout has reached sandbox evaluation and one optimizer step has completed.
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-1}"
export EXP_TAG="${EXP_TAG:-coding_agent_1node_smoke}"

export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-swebench}"

# The local E2B deployment uses host networking. E2B_DEBUG must remain false:
# true bypasses the API and targets a standalone envd on port 49983. Port 6379
# belongs to E2B Redis, so Ray uses 6380 by default.
export E2B_DEBUG="${E2B_DEBUG:-false}"
export E2B_API_URL="${E2B_API_URL:-http://127.0.0.1:3000}"
export E2B_SANDBOX_URL="${E2B_SANDBOX_URL:-http://127.0.0.1:3002}"
export E2B_PREFLIGHT="${E2B_PREFLIGHT:-1}"
export SLIME_AGENT_E2B_TEMPLATE_FROM_IMAGE="${SLIME_AGENT_E2B_TEMPLATE_FROM_IMAGE:-true}"
export RAY_PORT="${RAY_PORT:-6380}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export HOSTFILE="${HOSTFILE:-/dev/null}"
export MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-60000}"

exec bash "${SCRIPT_DIR}/run_qwen36_35b_a3b_swe_8nodes.sh" "$@"
