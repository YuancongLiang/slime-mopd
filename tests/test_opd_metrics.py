import json
from argparse import Namespace

import pytest
import torch

from slime.observability.opd_metrics import Collector, PipelineMetrics, aggregate, workload_metrics
from slime.utils.types import Sample

NUM_GPUS = 0


def test_rank_aggregation_counts_business_once_and_physical_bytes_all_ranks():
    records = [
        {
            "leader": leader,
            "counters": {
                "effective_tokens": tokens,
                "head_download_bytes": 100,
                "head_loss_rows": tokens,
                "learner_s": elapsed,
                "updates": 1,
                "domain/math/kl_token_sum": tokens * 2,
                "domain/math/kl_token_count": tokens,
            },
            "observations": {"teacher_queue_s": [elapsed]},
        }
        for leader, tokens, elapsed in [(True, 10, 1), (False, 10, 2), (True, 30, 3), (False, 30, 4)]
    ]
    result = aggregate(records)
    assert result["effective_tokens"] == result["head_loss_rows"] == 40
    assert result["head_download_bytes"] == 400
    assert result["learner_s"] == 4 and result["updates"] == 1
    assert result["domain/math/kl_token_mean"] == 2
    assert result["teacher_queue_s_count"] == 2
    assert result["teacher_queue_s_p95"] == pytest.approx(2.9)


def test_collector_defers_tensor_conversion_and_disabled_is_noop():
    collector = Collector()
    collector.add("ignored", 1)
    assert collector.snapshot()["counters"] == {}
    collector.enabled = True
    collector.add("sum", torch.tensor(2.0, requires_grad=True))
    collector.add("sum", torch.tensor(3.0))
    assert not collector.counters["sum"].requires_grad
    assert collector.snapshot()["counters"]["sum"] == 5


def test_workload_hash_ignores_targets_but_checks_masks_and_domains():
    sample = Sample(tokens=[1, 2, 3], response_length=2, loss_mask=[1, 0], metadata={"domain": "math"})
    before = workload_metrics([sample])
    sample.opd_target = {"teacher_id": "teacher"}
    sample.metadata["opd_metrics"] = {"target_wait_s": 0.1}
    assert workload_metrics([sample])["opd/workload_hash"] == before["opd/workload_hash"]
    assert before["opd/trajectory_tokens"] == 1 and before["opd/response_tokens"] == 2
    sample.remove_sample = True
    assert workload_metrics([sample])["opd/trajectory_tokens"] == 0
    assert workload_metrics([sample])["opd/workload_hash"] != before["opd/workload_hash"]


def test_pipeline_records_successful_counts_and_resumes_checkpoint_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr("slime.observability.logging_utils.log", lambda *a, **k: None)
    clock = iter([100.0, 110.0, 112.0, 113.0, 200.0, 210.0, 212.0, 213.0])
    monkeypatch.setattr("slime.observability.opd_metrics.time.perf_counter", lambda: next(clock))
    args = Namespace(
        use_opd=True,
        save=str(tmp_path),
        load="student",
        start_rollout_id=0,
        finetune=False,
        load_debug_rollout_data="replay.pt",
        opd_objective="sampled",
        rollout_temperature=0.8,
        opd_resolved={"student_temperature": 1.0},
    )
    learner = [{"effective_tokens": 100, "computed_tokens": 120, "learner_s": 0.5, "gpu": "test", "kl": float("nan")}]
    pipeline = PipelineMetrics(args)
    pipeline.begin()
    pipeline.finish(0, learner, {"opd/workload_hash": "same"})
    args.start_rollout_id = 1
    pipeline = PipelineMetrics(args)
    pipeline.begin()
    pipeline.finish(1, learner, {"opd/workload_hash": "same"})
    records = [json.loads(line) for line in (tmp_path / "opd_metrics.jsonl").read_text().splitlines()]
    assert records[0]["perf/opd_e2e_effective_tok_per_s"] == 50
    assert records[1]["perf/opd_total_effective_tokens"] == 200
    assert records[1]["perf/opd_total_train_s"] == 4
    assert records[0]["opd/computed_tokens"] == 120
    assert records[0]["opd/conditions"]["gpu"] == "test"
    assert records[0]["opd/conditions"]["student_temperature"] == 0.8
    assert records[0]["perf/opd_kl"] is None
