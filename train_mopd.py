"""Bounded three-pool MOPD coordinator with checkpoint-safe single-batch prefetch."""

import ray

from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.opd.config import configure, enabled
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.misc import should_run_periodic_action


def should_prefetch(config, rollout_id, num_rollout, checkpoint_due):
    # A checkpoint must not save a sampler cursor that has already consumed the next batch.
    return config["prefetch_rollouts"] and rollout_id + 1 < num_rollout and not checkpoint_due


def train(args):
    configure_logger()
    config = configure(args)
    if not enabled(args):
        raise ValueError("train_mopd.py requires --use-opd --opd-objective full_vocab_reverse_kl")
    if args.offload_train or args.offload_rollout or args.release_train:
        raise ValueError("Three-pool MOPD uses resident actors; offload/release are not supported by this coordinator")
    if args.update_weights_interval != 1:
        raise ValueError("The bounded MOPD coordinator publishes weights after every rollout batch")
    if args.rollout_function_path != "slime.rollout.sglang_rollout.generate_rollout":
        raise ValueError("This coordinator currently supports the standard completed-trajectory SGLang rollout")
    pgs = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor, _ = create_training_models(args, pgs, rollout_manager)
    actor.update_weights()
    pending = None
    try:
        if args.eval_interval is not None and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(args.start_rollout_id))
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            if pending is None:
                pending = rollout_manager.generate.remote(rollout_id)
            current = ray.get(pending)
            checkpoint_due = should_run_periodic_action(rollout_id, args.save_interval, per_epoch, args.num_rollout)
            pending = None
            if should_prefetch(config, rollout_id, args.num_rollout, checkpoint_due):
                pending = rollout_manager.generate.remote(rollout_id + 1)
            ray.get(actor.async_train(rollout_id, current))
            if checkpoint_due:
                actor.save_model(rollout_id, force_sync=True)
                ray.get(rollout_manager.save.remote(rollout_id))
            # Finish student generation before publishing; teacher prefill is frozen and may continue.
            if pending is not None:
                ray.get(pending)
            actor.update_weights()
            if should_run_periodic_action(rollout_id, args.eval_interval, per_epoch):
                ray.get(rollout_manager.eval.remote(rollout_id))
    finally:
        if pending is not None:
            ray.cancel(pending)
        ray.get(rollout_manager.dispose.remote())
        finish_tracking(args)


if __name__ == "__main__":
    train(parse_args())
