import copy
from argparse import Namespace

import httpx
import pytest
import torch
import yaml
from safetensors.torch import save

from slime.opd.config import configure, load_config
from slime.opd.protocol import read_tensor_response, request_id
from slime.opd.sampling import DomainSampler
from slime.opd.teacher_server import CapacityError, TargetStore, build_app

NUM_GPUS = 0


def config_file(tmp_path):
    cfg = {
        "schema_version": 1,
        "opd": {
            "backend": "torch",
            "teachers": {"a": {"version": "v1", "endpoints": ["http://localhost:7999"]}},
            "domains": {
                "math": {"teacher": "a", "sampling_weight": 0.25},
                "code": {"teacher": "a", "sampling_weight": 0.75},
            },
        },
    }
    path = tmp_path / "opd.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def test_config_preserves_legacy_defaults():
    args = Namespace(use_opd=False, loss_type="policy_loss")
    before = vars(args).copy()
    configure(args)
    assert vars(args) == before


@pytest.mark.parametrize("change", ["unknown", "negative", "nan", "empty", "missing_teacher", "bad_backend"])
def test_config_rejects_invalid_values(tmp_path, change):
    path, cfg = config_file(tmp_path)
    if change == "unknown":
        cfg["opd"]["chunk_token"] = 4
    elif change in {"negative", "nan"}:
        cfg["opd"]["domains"]["code"]["sampling_weight"] = -1 if change == "negative" else float("nan")
    elif change == "empty":
        cfg["opd"]["teachers"] = {}
    elif change == "missing_teacher":
        cfg["opd"]["domains"]["code"]["teacher"] = "absent"
    else:
        cfg["opd"]["backend"] = "sampled"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_config(path)


def test_config_enables_only_explicit_mode(tmp_path):
    path, _ = config_file(tmp_path)
    args = Namespace(
        use_opd=True,
        opd_objective="full_vocab_reverse_kl",
        opd_config=str(path),
        bf16=True,
        rollout_global_dataset=True,
    )
    configure(args)
    assert args.loss_type == "full_vocab_opd"
    assert not args.compute_advantages_and_returns
    args.pipeline_model_parallel_size = 2
    with pytest.raises(ValueError, match="pipeline"):
        configure(args)
    args.pipeline_model_parallel_size = 1
    args.partial_rollout = True
    with pytest.raises(ValueError, match="partial_rollout"):
        configure(args)


def test_weighted_sampling_and_resume(tmp_path):
    path, _ = config_file(tmp_path)
    cfg = load_config(path)
    samples = [Namespace(metadata={"domain": d}) for d in ("math", "math", "code", "code", "code")]
    sampler = DomainSampler(samples, cfg, 31)
    indices = sampler.draw(10000)
    assert 0.23 < sum(i < 2 for i in indices) / len(indices) < 0.27
    state = sampler.state_dict()
    expected = sampler.draw(200)
    restored = DomainSampler(samples, cfg, 31)
    restored.load_state_dict(state)
    assert restored.draw(200) == expected
    changed = copy.deepcopy(cfg)
    changed["sampling_fingerprint"] = "different"
    with pytest.raises(ValueError, match="changed"):
        DomainSampler(samples, changed, 31).load_state_dict(state)


def store_and_payload():
    info = {"teacher_id": "a", "version": "v1", "vocab_size": 20, "hidden_size": 8, "head_hash": "a" * 64}
    store = TargetStore(info, 1 << 20, 1, 100, 3600)
    payload = {"teacher_id": "a", "version": "v1", "tokens": [1, 2, 3, 4], "response_length": 2}
    payload["request_id"] = request_id("a", "v1", payload["tokens"], 2)
    return store, payload


def test_target_store_bounds_and_idempotency():
    store, payload = store_and_payload()
    key = store.submit(payload)
    assert store.submit(payload) == key
    assert len(store.entries) == 1
    other = payload | {"tokens": [1, 2, 3, 5]}
    other["request_id"] = request_id("a", "v1", other["tokens"], 2)
    with pytest.raises(CapacityError):
        store.submit(other)
    assert store.next()[0] == key
    with pytest.raises(ValueError):
        store.release(key)
    store.finish(key, b"data")
    assert store.get(key)["result"] == b"data"
    store.release(key)
    assert store.reserved == 0
    assert store.submit(other) == other["request_id"]


def test_hidden_http_roundtrip_and_padding():
    from slime.agent.aiohttp_threaded import run_app_in_thread

    store, payload = store_and_payload()
    head = torch.randn(20, 8, dtype=torch.bfloat16)
    hidden = torch.randn(2, 8, dtype=torch.bfloat16)
    handle = run_app_in_thread(build_app(store, head), host="127.0.0.1", port=0)
    url = f"http://127.0.0.1:{handle.port}"
    try:
        with httpx.Client(trust_env=False) as client:
            assert client.post(url + "/submit", json=payload).status_code == 202
            key, _ = store.next()
            assert client.get(url + "/result/" + key).status_code == 202
            store.finish(key, save({"hidden": hidden}))
            with client.stream("GET", url + "/result/" + key) as response:
                actual = read_tensor_response(response, "hidden", (2, 8), 65536)
            torch.testing.assert_close(actual, hidden)
            with client.stream("GET", url + "/head", params={"start": 15, "count": 10}) as response:
                shard = read_tensor_response(response, "head", (10, 8), 65536)
            torch.testing.assert_close(shard[:5], head[15:])
            assert not shard[5:].count_nonzero()
            assert client.delete(url + "/result/" + key).status_code == 204
            assert client.get(url + "/result/" + key).status_code == 404
    finally:
        handle.stop()


def test_policy_lag_counts_optimizer_steps():
    from slime.opd.megatron import validate_policy_versions

    validate_policy_versions([["4"]], {"4": 20}, 21, 1)
    with pytest.raises(ValueError, match="optimizer steps"):
        validate_policy_versions([["4"]], {"4": 20}, 23, 1)
    with pytest.raises(ValueError, match="exactly one"):
        validate_policy_versions([["4", "5"]], {"4": 20, "5": 21}, 21, 1)
    with pytest.raises(ValueError, match="expired"):
        validate_policy_versions([["3"]], {"4": 20}, 21, 1)


def test_checkpoint_boundary_does_not_prefetch():
    from train_mopd import should_prefetch

    config = {"prefetch_rollouts": True}
    assert should_prefetch(config, 0, 10, False)
    assert not should_prefetch(config, 0, 10, True)
    assert not should_prefetch(config, 9, 10, False)


def test_policy_clock_checkpoint_roundtrip(tmp_path):
    from slime.opd.state import load_state, save_state

    path, _ = config_file(tmp_path)
    cfg = load_config(path)
    heads = {"a": {"version": "v1", "hash": "a" * 64}}
    save_state(tmp_path, 9, cfg, 25, {"5": 24}, 5, heads)
    state = load_state(tmp_path, 9, cfg)
    assert load_state(None, -1, cfg) is None
    assert load_state(tmp_path, -1, cfg) is None
    assert state["optimizer_steps"] == 25 and state["version_steps"] == {"5": 24}
    assert state["teacher_heads"] == heads
    assert not (tmp_path / "rollout" / "opd_state_9.json.tmp").exists()
