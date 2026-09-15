#!/usr/bin/env bash
# Source with a default experiment group. Authentication uses wandb login or WANDB_API_KEY.
WANDB_ARGS=(
  --use-wandb
  --wandb-project "${WANDB_PROJECT:-slime-mopd}"
  --wandb-group "${WANDB_GROUP:-${WANDB_RUN_GROUP:-$1}}"
  --wandb-mode "${WANDB_MODE:-online}"
  --disable-wandb-random-suffix
)
if [[ -n ${WANDB_ENTITY:-} ]]; then
  WANDB_ARGS+=(--wandb-team "$WANDB_ENTITY")
fi
if [[ -n ${WANDB_RUN_ID:-} ]]; then
  WANDB_ARGS+=(--wandb-run-id "$WANDB_RUN_ID")
fi
if [[ -n ${WANDB_DIR:-} ]]; then
  WANDB_ARGS+=(--wandb-dir "$WANDB_DIR")
elif [[ -n ${SAVE_DIR:-} ]]; then
  WANDB_ARGS+=(--wandb-dir "$SAVE_DIR/wandb")
fi
