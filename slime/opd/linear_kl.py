"""Chunked linear reverse KL with bounded vocabulary scratch and TP support."""

import threading
import math
from contextlib import contextmanager

import torch
import torch.distributed as dist

from slime.observability.opd_metrics import get_collector


class Workspace:
    """One synchronous compute stream; autograd saves no references to scratch."""

    def __init__(self):
        self.buffers = {}
        self.lock = threading.Lock()

    @contextmanager
    def lease(self, hidden, weight, chunk):
        with self.lock:
            stream = torch.cuda.current_stream(hidden.device).cuda_stream if hidden.is_cuda else 0
            dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
            key = (hidden.device, hidden.dtype, dtype, weight.shape, chunk, stream)
            if key not in self.buffers:
                c, v, h = chunk, weight.shape[0], weight.shape[1]
                self.buffers[key] = {
                    "s": torch.empty(c, v, device=hidden.device, dtype=dtype),
                    "t": torch.empty(c, v, device=hidden.device, dtype=dtype),
                    "projection": torch.empty(c, v, device=hidden.device, dtype=hidden.dtype),
                    "grad_logits": torch.empty(c, v, device=hidden.device, dtype=hidden.dtype),
                    "input": torch.empty(c, h, device=hidden.device, dtype=hidden.dtype),
                }
                get_collector().add("workspace_allocations", 1)
                get_collector().add(
                    "workspace_allocated_bytes",
                    sum(t.numel() * t.element_size() for t in self.buffers[key].values()),
                )
            yield self.buffers[key]


def _project(x, weight, out, temporary, temperature):
    torch.mm(x, weight.t(), out=temporary)
    out.copy_(temporary).div_(temperature)


def _reduce(tensor, op, group):
    if group is not None and dist.get_world_size(group) > 1:
        with get_collector().time("kl_tp_reduce", gpu=tensor.is_cuda):
            dist.all_reduce(tensor, op=op, group=group)


def _stats(s, t, valid, group, backend):
    if valid == 0 and (group is None or dist.get_world_size(group) == 1):
        zeros = s.new_zeros(s.shape[0], dtype=torch.float32)
        return zeros, s.new_zeros((4, s.shape[0]), dtype=torch.float32)
    if backend == "tilelang":
        from .tilelang_kl import statistics

        return statistics(s, t, valid, group)
    s[:, valid:] = -torch.inf
    t[:, valid:] = -torch.inf
    maxima = torch.stack((s.max(-1).values, t.max(-1).values))
    _reduce(maxima, dist.ReduceOp.MAX, group)
    a, b = s - maxima[0, :, None], t - maxima[1, :, None]
    difference = a - b
    difference[:, valid:] = 0
    p = a.exp()
    moments = torch.stack((p.sum(-1), b.exp().sum(-1), (p * difference).sum(-1)))
    _reduce(moments, dist.ReduceOp.SUM, group)
    mean = moments[2] / moments[0]
    kl = mean - moments[0].log() + moments[1].log()
    return kl, torch.stack((maxima[0], maxima[1], moments[0], mean))


def _gradient(s, t, valid, stats, upstream, temperature, backend):
    if backend == "tilelang":
        from .tilelang_kl import gradient

        return gradient(s, t, valid, stats, upstream, temperature)
    a, b = s - stats[0, :, None], t - stats[1, :, None]
    s.copy_(a.exp() / stats[2, :, None] * (a - b - stats[3, :, None]))
    s[:, valid:] = 0
    s.mul_(upstream[:, None] / temperature)
    return s


class _LinearReverseKL(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        teacher_hidden,
        teacher_weight,
        vocab_size,
        group,
        chunk,
        ts,
        tt,
        backend,
        workspace,
        accumulate_main_grad,
    ):
        rank = dist.get_rank(group) if group is not None else 0
        valid = max(0, min(weight.shape[0], vocab_size - rank * weight.shape[0]))
        n = hidden.shape[0]
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        loss = torch.empty(n, dtype=dtype, device=hidden.device)
        saved = torch.empty(4, n, dtype=dtype, device=hidden.device)
        collector = get_collector()
        collector.add("head_loss_rows", n)
        with (
            collector.time("head_loss_forward", gpu=hidden.is_cuda),
            workspace.lease(hidden, weight, chunk) as scratch,
        ):
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                count = end - start
                s, t, projection = (scratch[k][:count] for k in ("s", "t", "projection"))
                _project(hidden[start:end], weight, s, projection, ts)
                _project(teacher_hidden[start:end], teacher_weight, t, projection, tt)
                loss[start:end], saved[:, start:end] = _stats(s, t, valid, group, backend)
        ctx.save_for_backward(hidden, weight, teacher_hidden, teacher_weight, saved)
        ctx.options = (valid, group, chunk, ts, tt, backend, workspace, accumulate_main_grad)
        return loss

    @staticmethod
    def backward(ctx, upstream):
        hidden, weight, th, tw, saved = ctx.saved_tensors
        valid, group, chunk, ts, tt, backend, workspace, main_grad = ctx.options
        dh = torch.empty_like(hidden)
        dw = weight.main_grad if main_grad else torch.zeros_like(weight)
        with (
            get_collector().time("head_loss_backward", gpu=hidden.is_cuda),
            workspace.lease(hidden, weight, chunk) as scratch,
        ):
            for start in range(0, hidden.shape[0], chunk):
                end = min(start + chunk, hidden.shape[0])
                count = end - start
                s, t, projection, dz = (scratch[k][:count] for k in ("s", "t", "projection", "grad_logits"))
                _project(hidden[start:end], weight, s, projection, ts)
                _project(th[start:end], tw, t, projection, tt)
                dz.copy_(_gradient(s, t, valid, saved[:, start:end], upstream[start:end], ts, backend))
                torch.mm(dz, weight, out=dh[start:end])
                if dw.dtype == dz.dtype:
                    dw.addmm_(dz.t(), hidden[start:end])
                else:
                    # Megatron's FP32 main_grad must not round through a BF16 weight-sized temporary.
                    from megatron.core.tensor_parallel.layers import fused_weight_gradient_mlp_cuda

                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(hidden[start:end], dz, dw)
            _reduce(dh, dist.ReduceOp.SUM, group)
        if main_grad:
            weight.grad_added_to_main_grad = True
            # Megatron DDP needs its parameter hook even though main_grad is already accumulated.
            from transformer_engine.pytorch.module.base import get_dummy_wgrad

            dw = get_dummy_wgrad(list(weight.shape), hidden.dtype, zero=getattr(weight, "zero_out_wgrad", False))
        return dh, dw, None, None, None, None, None, None, None, None, None, None


def fused_linear_reverse_kl(
    hidden,
    weight,
    teacher_hidden,
    teacher_weight,
    *,
    vocab_size=None,
    tp_group=None,
    chunk_tokens=256,
    student_temperature=1.0,
    teacher_temperature=1.0,
    backend="torch",
    workspace=None,
    accumulate_main_grad=False,
):
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("Expected hidden [rows,H] and head [local_vocab,H]")
    if teacher_hidden.shape != hidden.shape or teacher_weight.shape != weight.shape:
        raise ValueError("Student and teacher hidden/head shapes must agree")
    if any(x.device != hidden.device or x.dtype != hidden.dtype for x in (weight, teacher_hidden, teacher_weight)):
        raise ValueError("All linear KL inputs must share device and dtype")
    if teacher_hidden.requires_grad or teacher_weight.requires_grad:
        raise ValueError("Teacher tensors must be frozen")
    if (
        type(chunk_tokens) is not int
        or chunk_tokens < 1
        or any(not math.isfinite(t) or t <= 0 for t in (student_temperature, teacher_temperature))
    ):
        raise ValueError("Chunk size and temperatures must be positive")
    if backend not in {"torch", "tilelang"}:
        raise ValueError(f"Unknown linear KL backend: {backend}")
    if backend == "tilelang" and (not hidden.is_cuda or hidden.dtype != torch.bfloat16):
        raise ValueError("TileLang linear KL currently requires CUDA BF16 inputs")
    if backend == "tilelang" and torch.cuda.get_device_capability(hidden.device)[0] < 8:
        raise ValueError("TileLang linear KL requires an SM80 or newer GPU")
    size = dist.get_world_size(tp_group) if tp_group is not None else 1
    vocab_size = weight.shape[0] * size if vocab_size is None else vocab_size
    if not 0 < vocab_size <= weight.shape[0] * size:
        raise ValueError("Invalid global vocabulary size")
    if accumulate_main_grad and not hasattr(weight, "main_grad"):
        raise ValueError("Megatron main_grad buffer is required")
    return _LinearReverseKL.apply(
        hidden,
        weight,
        teacher_hidden,
        teacher_weight,
        vocab_size,
        tp_group,
        chunk_tokens,
        student_temperature,
        teacher_temperature,
        backend,
        workspace if workspace is not None else Workspace(),
        accumulate_main_grad,
    )
