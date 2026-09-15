import json

import pytest

from slime.opd.compare import compare_records, load_records


def records(objective, times=(1.0, 4.0)):
    return [
        {
            "opd/step": index,
            "opd/objective": objective,
            "opd/phase": "steady",
            "opd/batch_id": index,
            "opd/workload_hash": f"tokens-and-settings-{index}",
            "opd/benchmark_id": "fixed-run",
            "opd/conditions": {"tp": 1, "global_batch_size": 2, "teacher_identity": "teacher-v1"},
            "opd/effective_tokens": tokens,
            "opd/computed_tokens": tokens + 100,
            "perf/opd_learner_s": seconds,
            "perf/opd_e2e_s": seconds * 2,
            "perf/opd_peak_allocated_gib": 4 + index,
        }
        for index, (tokens, seconds) in enumerate(zip((100, 900), times, strict=True))
    ]


def test_comparison_uses_total_tokens_over_total_seconds():
    report = compare_records(records("sampled"), records("full_vocab_reverse_kl", (1.25, 5)))
    assert report["matched_workload"]
    assert report["sampled"]["learner"]["effective_tokens_per_second"] == 200
    assert report["comparison"]["learner"]["time_multiplier"] == 1.25
    assert report["comparison"]["learner"]["overhead_percent"] == 25
    assert report["comparison"]["learner"]["throughput_drop_percent"] == pytest.approx(20)
    assert report["sampled"]["peak_allocated_gib"] == 5


@pytest.mark.parametrize(
    "field", ["opd/workload_hash", "opd/effective_tokens", "opd/computed_tokens", "opd/benchmark_id", "opd/conditions"]
)
def test_mismatched_workloads_need_explicit_online_override(field):
    baseline, full = records("sampled"), records("full_vocab_reverse_kl")
    if field == "opd/conditions":
        full[0][field] = full[0][field] | {"tp": 2}
    else:
        full[0][field] = full[0][field] + 1 if isinstance(full[0][field], int) else "different"
    with pytest.raises(ValueError, match="Workloads differ"):
        compare_records(baseline, full)
    assert not compare_records(baseline, full, allow_different_workloads=True)["matched_workload"]


def test_warmup_excluded_and_optional_e2e_not_fabricated():
    baseline, full = records("sampled"), records("full_vocab_reverse_kl")
    for rows in (baseline, full):
        rows[0]["opd/phase"] = "warmup"
        del rows[1]["perf/opd_e2e_s"]
    report = compare_records(baseline, full)
    assert report["sampled"]["batches"] == 1
    assert "e2e" not in report["comparison"]


def test_gpu_sampling_uses_only_profiled_token_denominator():
    baseline, full = records("sampled"), records("full_vocab_reverse_kl")
    baseline[1]["perf/opd_head_loss_s"] = 1
    full[1]["perf/opd_head_loss_s"] = 1.5
    report = compare_records(baseline, full)
    assert report["sampled"]["head_loss"]["effective_tokens"] == 900
    assert report["comparison"]["head_loss"]["time_multiplier"] == 1.5


def test_gpu_comparison_rejects_different_profiled_batches():
    baseline, full = records("sampled"), records("full_vocab_reverse_kl")
    baseline[0]["perf/opd_head_loss_s"] = 1
    full[1]["perf/opd_head_loss_s"] = 1.5
    with pytest.raises(ValueError, match="Profiled batch IDs differ"):
        compare_records(baseline, full)


def test_duplicates_and_missing_hash_rejected():
    baseline, full = records("sampled"), records("full_vocab_reverse_kl")
    with pytest.raises(ValueError, match="Duplicate batch"):
        compare_records(baseline + baseline, full)
    for row in baseline + full:
        del row["opd/workload_hash"]
    with pytest.raises(ValueError, match="no workload hash"):
        compare_records(baseline, full)


def test_load_directory(tmp_path):
    (tmp_path / "opd_metrics.jsonl").write_text('{"opd/step": 1}\n\n')
    assert load_records(tmp_path) == [{"opd/step": 1}]


def test_resume_discards_uncheckpointed_tail_without_deleting_log_history(tmp_path):
    rows = records("sampled")
    resumed = rows[1] | {"opd/resume_from": 1, "perf/opd_learner_s": 5}
    path = tmp_path / "opd_metrics.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows + [resumed]))
    result = load_records(path)
    assert len(result) == 2 and result[1]["perf/opd_learner_s"] == 5
    assert len(path.read_text().splitlines()) == 3


def test_sampled_benchmark_uses_opd_policy_gradient_not_cross_entropy():
    import torch

    from slime.opd.benchmark import sampled_policy_loss

    logits = torch.tensor([[1.0, -0.5, 0.3], [-0.2, 0.4, 0.9]], requires_grad=True)
    labels = torch.tensor([0, 2])
    old_log_probs = logits.detach().log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
    teacher_log_probs = old_log_probs + torch.tensor([0.2, -0.4])
    probabilities = logits.detach().softmax(-1)
    loss, entropy = sampled_policy_loss(logits, labels, old_log_probs, teacher_log_probs, chunk_tokens=1)
    gradient = torch.autograd.grad(loss.mean(), logits)[0]
    probabilities[torch.arange(2), labels] -= 1
    expected = probabilities * (teacher_log_probs - old_log_probs)[:, None] / 2
    torch.testing.assert_close(gradient, expected)
    assert entropy.shape == (2,)
