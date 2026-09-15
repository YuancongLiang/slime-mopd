"""Instance-scoped head adapter and direct teacher-to-learner target loading."""

import time
from collections import OrderedDict
from contextlib import contextmanager
from types import MethodType

import torch
import torch.distributed as dist

from slime.observability.opd_metrics import get_collector

from .linear_kl import Workspace, fused_linear_reverse_kl
from .protocol import model_identity, read_tensor_response, request_id, verify_info


def unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def head_weight(model):
    model = unwrap(model)
    if model.share_embeddings_and_output_weights:
        return model.shared_embedding_or_output_weight()
    return model.output_layer.weight


@contextmanager
def head_forward(model, forward):
    layer = unwrap(model).output_layer
    previous = layer.__dict__.get("forward")
    layer.forward = MethodType(forward, layer)
    try:
        yield
    finally:
        if previous is None:
            del layer.forward
        else:
            layer.forward = previous


def gather_hidden(layer, hidden):
    if layer.sequence_parallel:
        from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

        hidden = gather_from_sequence_parallel_region(hidden, tensor_parallel_output_grad=False)
    return hidden


def validate_policy_versions(versions, published_steps, current_step, maximum_lag):
    if versions is None:
        raise ValueError("Missing student policy versions in MOPD batch")
    for sample_versions in versions:
        unique = {str(v) for v in sample_versions}
        if len(unique) != 1:
            raise ValueError("Each MOPD trajectory must contain exactly one student weight version")
        version = next(iter(unique))
        if version not in published_steps:
            raise ValueError(f"Unknown or expired student policy version {version}")
        age = current_step - published_steps[version]
        if not 0 <= age <= maximum_lag:
            raise ValueError(f"Student policy is {age} optimizer steps old; limit is {maximum_lag}")
        get_collector().observe("policy_lag_steps", age)


class Targets:
    def __init__(self, args):
        import httpx

        self.cfg = args.opd_resolved
        self.identity = model_identity(args.hf_checkpoint)
        self.client = httpx.Client(timeout=self.cfg["timeout_seconds"], trust_env=False)
        self.infos = {}
        self.teacher_hashes = {
            (name, item["version"]): item["hash"] for name, item in getattr(args, "_opd_teacher_heads", {}).items()
        }
        self.heads = OrderedDict()
        self.workspace = Workspace()

    def info(self, target):
        endpoints = [url.rstrip("/") for url in self.cfg["teachers"][target["teacher_id"]]["endpoints"]]
        if target["endpoint"] not in endpoints:
            # A checkpoint may move to new hosts without changing the frozen teacher identity.
            target["endpoint"] = endpoints[int(target["request_id"][:8], 16) % len(endpoints)]
        endpoint = target["endpoint"]
        if endpoint not in self.infos:
            response = self.client.get(endpoint + "/info")
            response.raise_for_status()
            self.infos[endpoint] = response.json()
        info = self.infos[endpoint]
        verify_info(info, target["teacher_id"], self.cfg["teachers"][target["teacher_id"]], self.identity)
        if target["version"] != info["version"]:
            raise ValueError("Queued target teacher version changed")
        key = (target["teacher_id"], target["version"])
        if self.teacher_hashes.setdefault(key, info["head_hash"]) != info["head_hash"]:
            raise ValueError("Teacher replicas advertise different heads for the same immutable version")
        return info

    def head(self, target, weight, rank):
        collector = get_collector()
        info = self.info(target)
        key = (target["teacher_id"], info["version"], info["head_hash"], rank, weight.shape[0])
        if key in self.heads:
            collector.add("head_cache_hits", 1)
            self.heads.move_to_end(key)
            return self.heads[key]
        collector.add("head_cache_misses", 1)
        count, width = weight.shape
        required = count * width * 2
        if required > self.cfg["head_cache_bytes"]:
            raise ValueError("One frozen head shard exceeds head_cache_bytes")
        started = time.perf_counter()
        with self.client.stream(
            "GET", target["endpoint"] + "/head", params={"start": rank * count, "count": count}
        ) as r:
            if r.headers.get("X-OPD-Head") != info["head_hash"]:
                raise ValueError("Teacher head changed during loading")
            head = read_tensor_response(r, "head", (count, width), required + 65536)
            collector.add("head_download_bytes", int(r.headers.get("content-length", required)))
        elapsed = time.perf_counter() - started
        collector.add("head_download_s", elapsed)
        collector.observe("head_download_latency_s", elapsed)
        # Evicted tensors remain alive when referenced by an in-flight autograd context.
        # PP=1 sequential forward/backward is required so this cache limit remains a memory bound.
        while (
            self.heads
            and sum(x.numel() * x.element_size() for x in self.heads.values()) + required
            > self.cfg["head_cache_bytes"]
        ):
            _, evicted = self.heads.popitem(last=False)
            collector.add("head_cache_evictions", 1)
            collector.add("head_cache_evicted_bytes", evicted.numel() * evicted.element_size())
        with collector.time("head_h2d", gpu=weight.is_cuda):
            head = head.to(device=weight.device, dtype=weight.dtype)
        self.heads[key] = head
        return head

    def hidden(self, target, tokens, response_length):
        collector = get_collector()
        info = self.info(target)
        if target["request_id"] != request_id(target["teacher_id"], target["version"], tokens, response_length):
            raise ValueError("Teacher target does not match this token trajectory")
        deadline = time.monotonic() + self.cfg["timeout_seconds"]
        endpoint = target["endpoint"]
        started = time.perf_counter()
        polls = 0
        while time.monotonic() < deadline:
            with self.client.stream("GET", endpoint + "/result/" + target["request_id"]) as response:
                polls += 1
                if response.status_code == 200:
                    collector.add("hidden_ready_hits", int(polls == 1))
                    collector.add("hidden_ready_misses", int(polls > 1))
                    collector.add("hidden_ready_wait_s", time.perf_counter() - started)
                    if response.headers.get("X-OPD-Head") != info["head_hash"]:
                        raise ValueError("Teacher hidden/head version mismatch")
                    shape = (response_length, self.identity["hidden_size"])
                    with collector.time("hidden_download"):
                        result = read_tensor_response(response, "hidden", shape, shape[0] * shape[1] * 2 + 65536)
                    collector.add(
                        "hidden_download_bytes", int(response.headers.get("content-length", result.numel() * 2))
                    )
                    for name in ("queue", "prefill", "d2h_wait", "serialize", "service"):
                        value = response.headers.get(f"X-OPD-{name.replace('_', '-')}-S")
                        if value is not None:
                            collector.observe(f"teacher_{name}_s", float(value))
                            collector.observe(f"teacher/{target['teacher_id']}/{name}_s", float(value))
                    peak = response.headers.get("X-OPD-Rank0-Peak-Allocated-Bytes")
                    if peak is not None:
                        collector.observe("teacher_rank0_peak_allocated_gib", int(peak) / 1024**3)
                    collector.add("teacher_input_tokens", len(tokens))
                    collector.add("teacher_response_tokens", response_length)
                    self.client.delete(endpoint + "/result/" + target["request_id"]).raise_for_status()
                    collector.observe("target_wait_latency_s", time.perf_counter() - started)
                    return result
                if response.status_code == 404:
                    collector.add("hidden_resubmits", 1)
                    retry = self.client.post(
                        endpoint + "/submit", json=target | {"tokens": tokens, "response_length": response_length}
                    )
                    if retry.status_code != 429:
                        retry.raise_for_status()
                    else:
                        collector.add("teacher_capacity_rejections", 1)
                elif response.status_code != 202:
                    response.raise_for_status()
            time.sleep(0.05)
        raise TimeoutError(f"Timed out waiting for teacher {target['teacher_id']}")


@contextmanager
def distillation_head(args, model, batch):
    from megatron.core import mpu

    if hasattr(args, "_opd_version_steps") and not getattr(args, "debug_train_only", False):
        validate_policy_versions(
            batch.get("weight_versions"),
            args._opd_version_steps,
            args._opd_optimizer_steps,
            args.opd_resolved["max_policy_lag_optimizer_steps"],
        )
    collector = get_collector()
    prepare_started = time.perf_counter()
    module = unwrap(model)
    if not hasattr(module, "_opd_targets"):
        module._opd_targets = Targets(args)
    runtime = module._opd_targets
    group = mpu.get_tensor_model_parallel_group()
    rank = mpu.get_tensor_model_parallel_rank()
    weight = head_weight(module)
    if module.output_layer.bias is not None:
        raise ValueError("Full-vocabulary OPD currently requires a bias-free LM head")
    if any(x is not None for x in (batch.get("multimodal_train_inputs"),)):
        raise ValueError("MOPD is text-only")
    targets = batch.get("opd_targets")
    if targets is None:
        raise ValueError("Missing teacher target descriptors")
    if any(not isinstance(t, dict) or t.get("teacher_id") not in runtime.cfg["teachers"] for t in targets):
        raise ValueError("Invalid teacher target descriptor")
    if (
        len({t["teacher_id"] for t in targets}) * weight.numel() * weight.element_size()
        > runtime.cfg["head_cache_bytes"]
    ):
        raise ValueError("Microbatch frozen heads exceed head_cache_bytes; increase budget or reduce packing")
    # A TP group must use identical requests and therefore the same collective sequence.
    signatures = [None] * dist.get_world_size(group)
    dist.all_gather_object(signatures, [t["request_id"] for t in targets], group=group)
    if any(s != signatures[0] for s in signatures):
        raise ValueError("Mismatched teacher requests within the learner TP group")
    hidden_parts, heads = [], {}
    for target, tokens, length in zip(targets, batch["unconcat_tokens"], batch["response_lengths"], strict=True):
        error, hidden = None, None
        try:
            heads[target["teacher_id"]] = runtime.head(target, weight, rank)
            if rank == 0:
                with collector.time("target_wait"):
                    hidden = runtime.hidden(target, tokens.cpu().tolist(), length)
                with collector.time("hidden_h2d", gpu=weight.is_cuda):
                    hidden = hidden.to(weight.device)
        except Exception as exc:
            error = str(exc)
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
        if any(errors):
            raise RuntimeError(f"MOPD target loading failed: {errors}")
        if rank != 0:
            hidden = torch.empty(length, weight.shape[1], dtype=weight.dtype, device=weight.device)
        with collector.time("hidden_tp_broadcast", gpu=weight.is_cuda):
            dist.broadcast(hidden, src=mpu.get_tensor_model_parallel_src_rank(), group=group)
        hidden_parts.append(hidden)
    collector.add("target_prepare_s", time.perf_counter() - prepare_started)
    collector.observe("head_cache_bytes", sum(t.numel() * t.element_size() for t in runtime.heads.values()))

    def forward(layer, input_, weight=None, **kwargs):
        hidden = gather_hidden(layer, input_).squeeze(1)
        output = hidden.new_zeros(hidden.shape[0], dtype=torch.float32)
        offset, grouped = 0, {}
        for target, th, total, response in zip(
            targets, hidden_parts, batch["total_lengths"], batch["response_lengths"], strict=True
        ):
            start, end = offset + total - response - 1, offset + total - 1
            group_rows, group_hidden = grouped.setdefault(target["teacher_id"], ([], []))
            group_rows.append(torch.arange(start, end, device=hidden.device))
            group_hidden.append(th)
            offset += total
        for teacher_id, (rows, teacher_parts) in grouped.items():
            rows = torch.cat(rows)
            values = fused_linear_reverse_kl(
                hidden.index_select(0, rows),
                weight if weight is not None else layer.weight,
                torch.cat(teacher_parts),
                heads[teacher_id],
                vocab_size=runtime.identity["vocab_size"],
                tp_group=group,
                chunk_tokens=runtime.cfg["chunk_tokens"],
                student_temperature=runtime.cfg["student_temperature"],
                teacher_temperature=runtime.cfg["teacher_temperature"],
                backend=runtime.cfg["backend"],
                workspace=runtime.workspace,
                accumulate_main_grad=True,
            )
            output = output.index_copy(0, rows, values)
        return output[:, None, None], None

    with head_forward(model, forward):
        yield
    collector.observe(
        "workspace_bytes",
        sum(t.numel() * t.element_size() for buffers in runtime.workspace.buffers.values() for t in buffers.values()),
    )


def loss(args, batch, output, sum_of_sample_mean):
    offset, parts = 0, []
    for total, response in zip(batch["total_lengths"], batch["response_lengths"], strict=True):
        parts.append(output[0, offset + total - response - 1 : offset + total - 1, 0])
        offset += total
    collector = get_collector()
    if collector.enabled:
        for part, mask, target in zip(parts, batch["loss_masks"], batch["opd_targets"], strict=True):
            kl_sum = (part.detach() * mask).sum()
            count = mask.sum()
            domain = target.get("domain", "unknown")
            for prefix in ("full_vocab_", f"domain/{domain}/"):
                collector.add(prefix + "kl_token_sum", kl_sum)
                collector.add(prefix + "kl_token_count", count)
                collector.add(prefix + "kl_sample_sum", kl_sum / count.clamp_min(1))
                collector.add(prefix + "kl_sample_count", count > 0)
    value = sum_of_sample_mean(torch.cat(parts))
    return value * args.opd_resolved["coefficient"], {"opd_reverse_kl": value.detach()}
