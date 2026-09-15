import asyncio
from argparse import Namespace

import pytest
import yaml

from slime.opd.replay import prepare
from slime.opd.sampled import configure, reward_func
from slime.utils.types import Sample

NUM_GPUS = 0


def test_sampled_domain_routing_preserves_original_reward_contract(tmp_path, monkeypatch):
    path = tmp_path / "domains.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "opd": {
                    "teachers": {"code_rl": {"version": "v1", "endpoints": ["http://code:8000"]}},
                    "domains": {"code": {"teacher": "code_rl", "sampling_weight": 1}},
                },
            }
        )
    )
    args = Namespace(use_opd=True, opd_objective="sampled", opd_type="sglang", opd_domain_config=str(path))
    configure(args)

    async def original(routed, sample, **kwargs):
        assert routed.rm_url == "http://code:8000/generate"
        return {"meta_info": {"input_token_logprobs": [[None], [-0.2], [-0.3]]}}

    monkeypatch.setattr("slime.opd.sampled.sampled_reward", original)
    sample = Sample(tokens=[1, 2, 3], response_length=1, metadata={"domain": "code"})
    result = asyncio.run(reward_func(args, sample))
    assert "meta_info" in result
    assert sample.metadata["opd_teacher_id"] == "code_rl"
    assert args.custom_reward_post_process_path.endswith("post_process_rewards")


@pytest.mark.parametrize(
    "objective,backend", [("sampled", "sglang"), ("sampled", "megatron"), ("full_vocab_reverse_kl", None)]
)
def test_replay_refreshes_targets_and_keeps_sglang_json_until_conversion(monkeypatch, objective, backend):
    async def teacher(args, sample, **kwargs):
        return {"meta_info": {"input_token_logprobs": []}}

    async def hidden(args, sample, **kwargs):
        sample.opd_target = {"request_id": "fresh"}
        return 0.0

    monkeypatch.setattr("slime.rollout.on_policy_distillation.reward_func", teacher)
    monkeypatch.setattr("slime.opd.rollout.reward_func", hidden)
    sample = Sample(
        tokens=[1, 2],
        response_length=1,
        reward={"meta_info": {"input_token_logprobs": []}},
        metadata={"opd_metrics": {"stale": 1}},
    )
    asyncio.run(prepare(Namespace(opd_objective=objective, opd_type=backend), [sample]))
    assert "opd_metrics" not in sample.metadata
    if backend == "sglang":
        assert "meta_info" in sample.reward
    else:
        assert sample.reward == 0.0
    if objective == "full_vocab_reverse_kl":
        assert sample.opd_target == {"request_id": "fresh"}


@pytest.mark.parametrize("reward", [2.0, {"score": 2.0}])
def test_megatron_target_refresh_preserves_task_rewards(reward):
    sample = Sample(tokens=[1, 2], response_length=1, reward=reward)
    asyncio.run(prepare(Namespace(opd_objective="sampled", opd_type="megatron"), [sample]))
    assert sample.reward == reward
