"""Domain routing for the existing SGLang sampled-token OPD reward hook."""

import asyncio
from copy import copy

from slime.opd.config import load_config
from slime.rollout.on_policy_distillation import reward_func as sampled_reward


def configure(args):
    if not (args.use_opd and args.opd_objective == "sampled" and args.opd_type == "sglang"):
        raise ValueError("--opd-domain-config requires sampled OPD with an SGLang teacher")
    args.opd_resolved = load_config(args.opd_domain_config)
    args.custom_rm_path = "slime.opd.sampled.reward_func"
    args.custom_reward_post_process_path = "slime.rollout.on_policy_distillation.post_process_rewards"


async def reward_func(args, sample, **kwargs):
    if isinstance(sample, list):
        return await asyncio.gather(*(reward_func(args, item, **kwargs) for item in sample))
    config = args.opd_resolved
    domain = sample.metadata[config["domain_key"]]
    teacher_id = config["domains"][domain]["teacher"]
    teacher = config["teachers"][teacher_id]
    routed_args = copy(args)
    endpoint = teacher["endpoints"][(sample.index or 0) % len(teacher["endpoints"])]
    routed_args.rm_url = endpoint.rstrip("/") + "/generate"
    sample.metadata["opd_teacher_id"] = teacher_id
    sample.metadata["opd_teacher_version"] = teacher["version"]
    return await sampled_reward(routed_args, sample, **kwargs)
