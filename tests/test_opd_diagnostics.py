"""Diagnostics follow actual target transfers and masked response losses."""

import time
from argparse import Namespace
from collections import OrderedDict, defaultdict
from contextlib import contextmanager

import httpx
import pytest
import torch
from safetensors.torch import save

from slime.opd import megatron
from slime.opd.protocol import request_id
from slime.opd.teacher_server import TargetStore, build_app

NUM_GPUS = 0


class RecordingCollector:
    enabled = True
    profile_gpu = False

    def __init__(self):
        self.counters = defaultdict(float)
        self.observations = defaultdict(list)

    def add(self, name, value):
        self.counters[name] += float(value)

    def observe(self, name, value):
        self.observations[name].append(value)

    @contextmanager
    def time(self, name, gpu=False):
        started = time.perf_counter()
        yield
        self.add(name + "_s", time.perf_counter() - started)


def test_domain_kl_ignores_masked_positions(monkeypatch):
    collector = RecordingCollector()
    monkeypatch.setattr(megatron, "get_collector", lambda: collector)
    output = torch.tensor([[[0.0], [2.0], [1000.0], [4.0], [0.0], [0.0], [6.0], [0.0]]])
    batch = {
        "total_lengths": [5, 3],
        "response_lengths": [3, 1],
        "loss_masks": [torch.tensor([1, 0, 1]), torch.tensor([1])],
        "opd_targets": [{"domain": "math"}, {"domain": "code"}],
    }
    loss, _ = megatron.loss(Namespace(opd_resolved={"coefficient": 2}), batch, output, lambda x: x.sum())
    assert loss == 2024
    assert collector.counters["full_vocab_kl_token_sum"] == 12
    assert collector.counters["full_vocab_kl_token_count"] == 3
    assert collector.counters["domain/math/kl_sample_sum"] == 3
    assert collector.counters["domain/math/kl_token_count"] == 2
    assert collector.counters["domain/code/kl_sample_sum"] == 6


def test_teacher_transfer_and_cache_metrics(monkeypatch):
    from slime.agent.aiohttp_threaded import run_app_in_thread

    collector = RecordingCollector()
    monkeypatch.setattr(megatron, "get_collector", lambda: collector)
    identity = {"model_hash": "model", "tokenizer_hash": "tokenizer", "hidden_size": 8, "vocab_size": 20}
    info = identity | {"protocol": 1, "teacher_id": "teacher", "version": "v1", "head_hash": "a" * 64}
    store = TargetStore(info, 1 << 20, 4, 64, 3600)
    head, hidden = torch.randn(20, 8, dtype=torch.bfloat16), torch.randn(2, 8, dtype=torch.bfloat16)
    handle = run_app_in_thread(build_app(store, head), host="127.0.0.1", port=0)
    endpoint = f"http://127.0.0.1:{handle.port}"
    tokens = [1, 2, 3, 4]
    target = {
        "teacher_id": "teacher",
        "version": "v1",
        "endpoint": endpoint,
        "request_id": request_id("teacher", "v1", tokens, 2),
    }
    key = store.submit(target | {"tokens": tokens, "response_length": 2})
    store.next()
    store.finish(key, save({"hidden": hidden}), metrics={"prefill_s": 0.125, "serialize_s": 0.002})
    runtime = megatron.Targets.__new__(megatron.Targets)
    runtime.cfg = {
        "teachers": {"teacher": {"version": "v1", "endpoints": [endpoint]}},
        "head_cache_bytes": 1 << 20,
        "timeout_seconds": 5,
    }
    runtime.identity, runtime.infos, runtime.teacher_hashes, runtime.heads = identity, {}, {}, OrderedDict()
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            runtime.client = client
            first = runtime.head(target, head, 0)
            assert runtime.head(target, head, 0) is first
            torch.testing.assert_close(runtime.hidden(target, tokens, 2), hidden)
            stats = client.get(endpoint + "/loads").json()
        assert collector.counters["head_cache_hits"] == collector.counters["head_cache_misses"] == 1
        assert collector.counters["head_download_bytes"] > head.numel() * head.element_size()
        assert collector.counters["hidden_download_bytes"] > hidden.numel() * hidden.element_size()
        assert collector.counters["hidden_ready_hits"] == 1
        assert collector.observations["teacher_prefill_s"] == [0.125]
        assert collector.observations["teacher/teacher/serialize_s"] == [0.002]
        assert stats["counters"]["input_tokens"] == 4
        assert stats["counters"]["response_tokens"] == 2
        assert stats["timings"]["prefill_s"] == {"sum": 0.125, "count": 1, "recent": [0.125]}
        assert stats["reserved_bytes"] == 0
    finally:
        handle.stop()


def test_teacher_recent_latencies_are_bounded():
    info = {"teacher_id": "a", "version": "v1", "vocab_size": 20, "hidden_size": 8}
    store = TargetStore(info, 1 << 20, 1, 64, 3600)
    payload = {"teacher_id": "a", "version": "v1", "tokens": [1, 2, 3], "response_length": 1}
    payload["request_id"] = request_id("a", "v1", payload["tokens"], 1)
    for _ in range(300):
        key = store.submit(payload)
        store.next()
        store.finish(key, b"hidden", metrics={"prefill_s": 0.01})
        store.release(key)
    metric = store.stats()["timings"]["prefill_s"]
    assert metric["sum"] == pytest.approx(3)
    assert metric["count"] == 300
    assert len(metric["recent"]) == 256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA timing requires a GPU")
def test_profiled_linear_kl_records_both_autograd_phases(monkeypatch):
    from slime.observability.opd_metrics import Collector
    from slime.opd import linear_kl

    collector = Collector()
    collector.enabled = collector.profile_gpu = True
    monkeypatch.setattr(linear_kl, "get_collector", lambda: collector)
    hidden = torch.randn(6, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    linear_kl.fused_linear_reverse_kl(
        hidden, weight, torch.randn_like(hidden), torch.randn_like(weight), chunk_tokens=4
    ).sum().backward()
    assert len(collector.events) == 2
    counters = collector.snapshot()["counters"]
    assert counters["head_loss_forward_s"] > 0
    assert counters["head_loss_backward_s"] > 0
    assert counters["head_loss_rows"] == 6
    assert counters["workspace_allocations"] == 1
