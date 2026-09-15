"""Small, batch-scoped OPD counters and matched-workload performance records."""

import hashlib
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class Collector:
    def __init__(self):
        self.enabled = False
        self.profile_gpu = False
        self.counters = {}
        self.observations = defaultdict(list)
        self.events = []

    def start(self, args, batch_id):
        self.__init__()
        self.enabled = getattr(args, "use_opd", False)
        interval = getattr(args, "opd_metrics_interval", 20)
        self.profile_gpu = self.enabled and interval > 0 and batch_id % interval == 0
        if self.enabled:
            import torch

            torch.cuda.reset_peak_memory_stats()

    def add(self, name, value):
        if self.enabled:
            if hasattr(value, "detach"):
                value = value.detach()
            self.counters[name] = self.counters.get(name, 0) + value

    def observe(self, name, value):
        if self.enabled:
            self.observations[name].append(value)

    @contextmanager
    def time(self, name, gpu=False):
        if not self.enabled or (gpu and not self.profile_gpu):
            yield
            return
        if gpu:
            import torch

            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self.events.append((name + "_s", start, end))
        else:
            start = time.perf_counter()
            try:
                yield
            finally:
                self.add(name + "_s", time.perf_counter() - start)

    def snapshot(self):
        # Events are read once at the batch boundary, never synchronized per token/chunk.
        for name, start, end in self.events:
            end.synchronize()
            self.add(name, start.elapsed_time(end) / 1000)
        self.events.clear()
        return {
            "counters": {k: float(v) for k, v in self.counters.items()},
            "observations": dict(self.observations),
        }


_COLLECTOR = Collector()


def get_collector():
    return _COLLECTOR


def aggregate(records):
    """Business counts use TP/CP leaders; physical downloads include every shard."""
    counters, observations = defaultdict(list), defaultdict(list)
    for record in records:
        for key, value in record["counters"].items():
            physical = key.startswith(("head_cache_", "head_download_"))
            if record["leader"] or physical or key.endswith(("_s", "_gib")):
                counters[key].append(value)
        for key, values in record["observations"].items():
            if record["leader"] or key.startswith("head_"):
                observations[key].extend(values)
    result = {}
    for key, values in counters.items():
        result[key] = (
            max(values) if key.endswith(("_s", "_gib")) or key in {"updates", "skipped_updates"} else sum(values)
        )
    for key, values in observations.items():
        result.update(
            {
                f"{key}_{stat}": float(value)
                for stat, value in {
                    "count": len(values),
                    "mean": np.mean(values),
                    "p50": np.percentile(values, 50),
                    "p95": np.percentile(values, 95),
                    "max": max(values),
                }.items()
            }
        )
    for key in list(result):
        if key.endswith("_sum"):
            count = result.get(key[:-4] + "_count", 0)
            if count:
                result[key[:-4] + "_mean"] = result[key] / count
    return result


def finish_learner(args):
    import torch
    import torch.distributed as dist
    from megatron.core import mpu

    from slime.utils.distributed_utils import get_gloo_group

    collector = get_collector()
    if not collector.enabled:
        return None
    collector.add("peak_allocated_gib", torch.cuda.max_memory_allocated() / 1024**3)
    collector.add("peak_reserved_gib", torch.cuda.max_memory_reserved() / 1024**3)
    record = collector.snapshot()
    if "head_loss_backward_s" in record["counters"]:
        record["counters"]["head_loss_s"] = sum(
            record["counters"].get(key + "_s", 0) for key in ("head_loss_forward", "head_loss_backward")
        )
    if "learner_backward_s" in record["counters"]:
        record["counters"]["learner_compute_s"] = sum(
            record["counters"].get(key + "_s", 0)
            for key in ("learner_forward", "learner_loss", "learner_backward", "learner_finalize", "optimizer")
        )
    record["leader"] = (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_context_parallel_rank() == 0
        and mpu.is_pipeline_last_stage(ignore_virtual=True)
    )
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, record, group=get_gloo_group())
    collector.enabled = False
    if dist.get_rank() == 0:
        result = aggregate(records)
        accesses = result.get("head_cache_hits", 0) + result.get("head_cache_misses", 0)
        if accesses:
            result["head_cache_hit_ratio"] = result.get("head_cache_hits", 0) / accesses
        result["gpu"] = torch.cuda.get_device_name()
        return result
    return None


def workload_metrics(samples, domain_key="domain"):
    digest = hashlib.sha256()
    effective, computed = 0, 0
    observations = defaultdict(list)
    domains = defaultdict(lambda: [0, 0])
    for sample in samples:
        mask = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
        if sample.remove_sample:
            mask = [0] * sample.response_length
        domain = (sample.metadata or {}).get(domain_key, "unknown")
        item = [sample.tokens, sample.response_length, mask, domain]
        digest.update(json.dumps(item, separators=(",", ":")).encode())
        digest.update(b"\n")
        effective += sum(value > 0 for value in mask)
        computed += sample.response_length
        domains[domain][0] += 1
        domains[domain][1] += sum(value > 0 for value in mask)
        for key, value in (sample.metadata or {}).get("opd_metrics", {}).items():
            observations[key].append(value)
    result = {
        "opd/workload_hash": digest.hexdigest(),
        "opd/trajectory_tokens": effective,
        "opd/response_tokens": computed,
        "opd/samples": len(samples),
    } | {f"perf/opd_{key}_mean": sum(values) / len(values) for key, values in observations.items()}
    for domain, (count, tokens) in domains.items():
        result[f"opd/domain/{domain}/sample_share"] = count / len(samples)
        result[f"opd/domain/{domain}/token_share"] = tokens / effective if effective else 0
    return result


class PipelineMetrics:
    def __init__(self, args):
        self.args = args
        self.enabled = getattr(args, "use_opd", False)
        self.elapsed = 0.0
        self.tokens = 0
        self.started = time.perf_counter()
        self.restored_job_seconds = 0.0
        self.restored = False
        self.first_record = True

    def begin(self):
        if self.enabled and not self.restored:
            directory = getattr(self.args, "opd_metrics_dir", None) or self.args.save
            path = Path(directory) / "opd_metrics.jsonl" if directory else None
            if path and path.exists() and not self.args.finetune:
                with path.open() as stream:
                    for line in stream:
                        record = json.loads(line)
                        if record["opd/batch_id"] < self.args.start_rollout_id:
                            self.tokens = record["perf/opd_total_effective_tokens"]
                            self.elapsed = record.get("perf/opd_total_train_s", 0)
                            self.restored_job_seconds = record.get("perf/opd_job_elapsed_s", 0)
            self.restored = True
        self.start = time.perf_counter()

    def finish(self, batch_id, learner_results, workload, excluded_s=0.0):
        if not self.enabled:
            return
        from slime.observability.logging_utils import log

        seconds = time.perf_counter() - self.start - excluded_s
        learner = next((r for r in learner_results if isinstance(r, dict) and "effective_tokens" in r), {})
        effective = learner.get("effective_tokens", 0)
        objective = getattr(self.args, "opd_objective", "sampled")
        config = getattr(self.args, "opd_resolved", {})
        student_temperature = (
            config.get("student_temperature", 1.0)
            if objective == "full_vocab_reverse_kl"
            else getattr(self.args, "rollout_temperature", 1.0)
        )
        teacher_temperature = (
            config.get("teacher_temperature", 1.0)
            if objective == "full_vocab_reverse_kl"
            else student_temperature
            if getattr(self.args, "opd_type", None) == "megatron"
            else 1.0
        )
        self.elapsed += seconds
        self.tokens += effective
        record = dict(workload)
        record.update({f"perf/opd_{key}": value for key, value in learner.items() if key != "gpu"})
        record.update(
            {
                "opd/step": batch_id,
                "opd/batch_id": batch_id,
                "opd/objective": objective,
                "opd/phase": "warmup"
                if batch_id - self.args.start_rollout_id < getattr(self.args, "opd_metrics_warmup", 2)
                else "steady",
                "opd/effective_tokens": effective,
                "opd/warmup": int(batch_id - self.args.start_rollout_id < getattr(self.args, "opd_metrics_warmup", 2)),
                "opd/computed_tokens": learner.get("computed_tokens", 0),
                "opd/benchmark_id": getattr(self.args, "opd_benchmark_id", None),
                "opd/gpu": learner.get("gpu"),
                "opd/replay": bool(self.args.load_debug_rollout_data),
                "opd/conditions": {
                    "gpu": learner.get("gpu"),
                    "student_checkpoint": self.args.load,
                    "teacher_identity": getattr(self.args, "opd_benchmark_teacher_id", None),
                    "student_temperature": student_temperature,
                    "teacher_temperature": teacher_temperature,
                    "prefetch_rollouts": config.get("prefetch_rollouts", False)
                    if objective == "full_vocab_reverse_kl"
                    else False,
                }
                | {
                    key: getattr(self.args, key, None)
                    for key in (
                        "global_batch_size",
                        "tensor_model_parallel_size",
                        "pipeline_model_parallel_size",
                        "context_parallel_size",
                        "expert_model_parallel_size",
                        "actor_num_nodes",
                        "actor_num_gpus_per_node",
                        "rollout_num_gpus",
                        "bf16",
                        "micro_batch_size",
                        "use_dynamic_batch_size",
                        "max_tokens_per_gpu",
                        "balance_data",
                        "balance_by_flops",
                        "log_probs_chunk_size",
                        "log_probs_max_tokens_per_gpu",
                        "use_rollout_logprobs",
                        "recompute_granularity",
                        "recompute_method",
                        "recompute_num_layers",
                        "use_distributed_optimizer",
                        "sequence_parallel",
                        "opd_metrics_interval",
                    )
                },
                "perf/opd_e2e_s": seconds,
                "perf/opd_e2e_effective_tok_per_s": effective / seconds,
                "perf/opd_total_effective_tokens": self.tokens,
                "perf/opd_total_train_s": self.elapsed,
                "perf/opd_job_elapsed_s": self.restored_job_seconds + time.perf_counter() - self.started,
            }
        )
        if self.first_record:
            record["opd/resume_from"] = self.args.start_rollout_id
            self.first_record = False
        record["perf/opd_job_effective_tok_per_s"] = self.tokens / record["perf/opd_job_elapsed_s"]
        if effective:
            record["perf/opd_e2e_ms_per_1k_tokens"] = seconds * 1e6 / effective
        for scope in ("learner", "learner_compute"):
            if learner.get(scope + "_s", 0) > 0 and effective:
                record[f"perf/opd_{scope}_ms_per_1k_tokens"] = learner[scope + "_s"] * 1e6 / effective
        record = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in record.items()}
        log(self.args, {k: v for k, v in record.items() if isinstance(v, (int, float))}, step_key="opd/step")
        directory = getattr(self.args, "opd_metrics_dir", None) or self.args.save
        if directory:
            path = Path(directory) / "opd_metrics.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
