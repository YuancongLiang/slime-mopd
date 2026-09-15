"""Refresh frozen-teacher targets when replaying saved student trajectories."""

import asyncio


async def prepare(args, samples):
    for sample in samples:
        sample.metadata.pop("opd_metrics", None)
    if args.opd_objective == "full_vocab_reverse_kl":
        from slime.opd.rollout import reward_func

        rewards = await asyncio.gather(*(reward_func(args, sample) for sample in samples))
        for sample, reward in zip(samples, rewards, strict=True):
            sample.reward = reward
    elif args.opd_type == "sglang":
        from slime.rollout.on_policy_distillation import reward_func

        if getattr(args, "opd_domain_config", None):
            from slime.opd.sampled import reward_func

        rewards = await asyncio.gather(*(reward_func(args, sample) for sample in samples))
        for sample, reward in zip(samples, rewards, strict=True):
            sample.reward = reward
    else:
        # Pure distillation replay can originate from an SGLang dump with a JSON reward.
        for sample in samples:
            if isinstance(sample.reward, dict) and "input_token_logprobs" in sample.reward.get("meta_info", {}):
                sample.reward = 0.0
            sample.teacher_log_probs = None
