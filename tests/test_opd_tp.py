from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from slime.opd.linear_kl import fused_linear_reverse_kl

NUM_GPUS = 0


def _worker(rank, world, rendezvous, valid, cuda):
    if cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if cuda else "gloo", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=180)
    )
    try:
        torch.manual_seed(12)
        device, dtype = (f"cuda:{rank}", torch.bfloat16) if cuda else ("cpu", torch.float64)
        h = torch.randn(11, 32, device=device, dtype=dtype, requires_grad=True)
        full = torch.randn(1024, 32, device=device, dtype=dtype, requires_grad=True)
        th, tw = torch.randn_like(h), torch.randn_like(full)
        local = full.detach().chunk(world)[rank].contiguous().requires_grad_()
        teacher_local = tw.chunk(world)[rank].contiguous()
        p = ((h @ full[:valid].t()).float() if cuda else h @ full[:valid].t()).log_softmax(-1)
        q = ((th @ tw[:valid].t()).float() if cuda else th @ tw[:valid].t()).log_softmax(-1)
        expected = (p.exp() * (p - q)).sum(-1)
        dh, dw = torch.autograd.grad(expected.sum(), (h, full))
        actual = fused_linear_reverse_kl(
            h,
            local,
            th,
            teacher_local,
            tp_group=dist.group.WORLD,
            vocab_size=valid,
            chunk_tokens=4,
            backend="tilelang" if cuda else "torch",
        )
        actual_dh, actual_dw = torch.autograd.grad(actual.sum(), (h, local))
        tolerance = dict(atol=0.063, rtol=0.04) if cuda else {}
        torch.testing.assert_close(actual, expected, **tolerance)
        torch.testing.assert_close(actual_dh, dh, **tolerance)
        torch.testing.assert_close(actual_dw, dw.chunk(world)[rank], **tolerance)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("valid", [997, 403])
def test_tp_reference_with_padding(tmp_path, valid):
    mp.spawn(_worker, args=(2, "file://" + str(tmp_path / "rendezvous"), valid, False), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA GPUs required")
def test_tilelang_tp_with_wholly_padded_rank(tmp_path):
    mp.spawn(_worker, args=(2, "file://" + str(tmp_path / "rendezvous"), 403, True), nprocs=2, join=True)
