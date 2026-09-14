"""Real CUDA/Megatron head tests without downloading a backbone checkpoint."""

import os
from argparse import Namespace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

NUM_GPUS = 2


def _head_worker(rank, rendezvous, sp):
    torch.cuda.set_device(rank)
    from megatron.core import mpu
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig
    from slime.opd.linear_kl import Workspace, fused_linear_reverse_kl
    from slime.opd.megatron import gather_hidden, head_forward

    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    mpu.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            sequence_parallel=sp,
            params_dtype=torch.bfloat16,
            bf16=True,
            use_cpu_initialization=True,
        )
        torch.manual_seed(44)
        layer = ColumnParallelLinear(32, 128, config=cfg, init_method=torch.nn.init.normal_, bias=False).cuda()
        full_weight = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16)
        teacher_weight = torch.randn_like(full_weight)
        full_hidden = torch.randn(8, 1, 32, device="cuda", dtype=torch.bfloat16)
        teacher_hidden = torch.randn_like(full_hidden)
        layer.weight.data.copy_(full_weight.chunk(2)[rank])
        layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)
        layer.weight.grad_added_to_main_grad = False
        layer.weight.zero_out_wgrad = True
        local_h = (full_hidden.chunk(2)[rank] if sp else full_hidden).clone().requires_grad_()
        model = Namespace(output_layer=layer)
        original_parameter = layer.weight
        original_state = set(layer.state_dict())

        def forward(head, input_, weight=None, **kwargs):
            h = gather_hidden(head, input_).squeeze(1)
            out = fused_linear_reverse_kl(
                h,
                head.weight,
                teacher_hidden.squeeze(1),
                teacher_weight.chunk(2)[rank].contiguous(),
                tp_group=mpu.get_tensor_model_parallel_group(),
                backend="tilelang",
                chunk_tokens=4,
                workspace=Workspace(),
                accumulate_main_grad=True,
            )
            return out, None

        with head_forward(model, forward):
            actual = layer(local_h)[0]
        actual.sum().backward()
        assert layer.weight is original_parameter
        assert set(layer.state_dict()) == original_state
        assert "forward" not in layer.__dict__
        assert layer.weight.grad_added_to_main_grad
        ref_h = full_hidden.squeeze(1).clone().requires_grad_()
        ref_w = full_weight.clone().requires_grad_()
        p = (ref_h @ ref_w.t()).float().log_softmax(-1)
        q = (teacher_hidden.squeeze(1) @ teacher_weight.t()).float().log_softmax(-1)
        expected = (p.exp() * (p - q)).sum(-1)
        dh, dw = torch.autograd.grad(expected.sum(), (ref_h, ref_w))
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(local_h.grad.squeeze(1), dh.chunk(2)[rank] if sp else dh, atol=0.063, rtol=0.04)
        torch.testing.assert_close(layer.weight.main_grad, dw.chunk(2)[rank].float(), atol=0.063, rtol=0.04)
        assert not layer.weight.grad.count_nonzero()
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA GPUs required")
@pytest.mark.parametrize("sp", [False, True])
def test_megatron_main_grad_and_sp(tmp_path, sp):
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    mp.spawn(_head_worker, args=("file://" + str(tmp_path / "rendezvous"), sp), nprocs=2, join=True)
