"""Tiny tied-embedding GPT: HTTP teacher hidden -> MOPD head -> Megatron DDP gradients."""

from argparse import Namespace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

NUM_GPUS = 1


def _step_worker(rank, rendezvous, tied):
    torch.cuda.set_device(0)
    from megatron.core import mpu, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.transformer_config import TransformerConfig
    from safetensors.torch import save

    from slime.agent.aiohttp_threaded import run_app_in_thread
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.data import DataIterator, get_batch
    from slime.opd.linear_kl import Workspace
    from slime.opd.megatron import Targets, distillation_head, gather_hidden, head_forward, head_weight, loss
    from slime.opd.protocol import request_id
    from slime.opd.teacher_server import TargetStore, build_app

    dist.init_process_group("nccl", init_method=rendezvous, rank=0, world_size=1, timeout=timedelta(seconds=120))
    mpu.initialize_model_parallel()
    tensor_parallel.model_parallel_cuda_manual_seed(123)
    handle = None
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=128,
            params_dtype=torch.bfloat16,
            bf16=True,
            hidden_dropout=0,
            attention_dropout=0,
            gradient_accumulation_fusion=False,
            add_bias_linear=False,
        )
        spec = get_gpt_layer_with_transformer_engine_spec()

        def make_model():
            return (
                GPTModel(
                    config,
                    spec,
                    vocab_size=128,
                    max_sequence_length=64,
                    share_embeddings_and_output_weights=tied,
                    position_embedding_type="rope",
                )
                .cuda()
                .bfloat16()
            )

        student, reference, teacher = make_model(), make_model(), make_model()
        reference.load_state_dict(student.state_dict())
        teacher.requires_grad_(False)
        teacher.eval()
        data = {
            "tokens": [torch.tensor([1, 2, 3, 4, 5, 6], device="cuda"), torch.tensor([7, 8, 9, 10], device="cuda")],
            "total_lengths": [6, 4],
            "response_lengths": [3, 2],
            "loss_masks": [torch.tensor([1, 1, 0], device="cuda"), torch.tensor([1, 1], device="cuda")],
            "rollout_mask_sums": [torch.tensor(2, device="cuda"), torch.tensor(2, device="cuda")],
        }
        batch = get_batch(DataIterator(data, [[0, 1]]), list(data), pad_multiplier=16)
        kwargs = dict(
            input_ids=batch["tokens"],
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=batch["packed_seq_params"],
            loss_mask=batch["full_loss_masks"],
        )

        def capture(layer, input_, **kw):
            return gather_hidden(layer, input_), None

        with torch.inference_mode(), head_forward(teacher, capture):
            hidden = teacher(**kwargs)
        teacher_head = head_weight(teacher).detach().cpu().contiguous()
        identity = {"model_hash": "model", "tokenizer_hash": "tokenizer", "hidden_size": 64, "vocab_size": 128}
        info = identity | {"protocol": 1, "teacher_id": "teacher", "version": "v1", "head_hash": "a" * 64}
        store = TargetStore(info, 1 << 20, 8, 64, 3600)
        handle = run_app_in_thread(build_app(store, teacher_head), host="127.0.0.1", port=0)
        endpoint = f"http://127.0.0.1:{handle.port}"
        targets, offset = [], 0
        for tokens, total, response in zip(
            data["tokens"], data["total_lengths"], data["response_lengths"], strict=True
        ):
            target = {
                "teacher_id": "teacher",
                "version": "v1",
                "endpoint": endpoint,
                "request_id": request_id("teacher", "v1", tokens.tolist(), response),
            }
            store.submit(target | {"tokens": tokens.tolist(), "response_length": response})
            key, _ = store.next()
            th = hidden[0, offset + total - response - 1 : offset + total - 1].cpu().contiguous()
            store.finish(key, save({"hidden": th}))
            targets.append(target)
            offset += total
        batch["opd_targets"] = targets
        cfg = {
            "teachers": {"teacher": {"version": "v1", "endpoints": [endpoint]}},
            "backend": "tilelang",
            "head_cache_bytes": 1 << 20,
            "chunk_tokens": 4,
            "student_temperature": 1.0,
            "teacher_temperature": 1.0,
            "coefficient": 1.0,
            "timeout_seconds": 10,
        }
        import httpx

        runtime = Targets.__new__(Targets)
        runtime.cfg, runtime.identity = cfg, identity
        runtime.client = httpx.Client(timeout=10, trust_env=False)
        runtime.infos, runtime.teacher_hashes = {}, {}
        from collections import OrderedDict

        runtime.heads = OrderedDict()
        runtime.workspace = Workspace()
        student._opd_targets = runtime
        wrapped = DistributedDataParallel(config, DistributedDataParallelConfig(grad_reduce_in_fp32=True), student)
        wrapped.zero_grad_buffer()
        reducer = get_sum_of_sample_mean(
            data["total_lengths"], data["response_lengths"], data["loss_masks"], data["rollout_mask_sums"]
        )
        args = Namespace(opd_resolved=cfg)
        before = set(student.state_dict())
        with distillation_head(args, wrapped, batch):
            output = wrapped(**kwargs)
        actual, _ = loss(args, batch, output, reducer)
        actual.backward()
        wrapped.finish_grad_sync()
        assert set(student.state_dict()) == before
        ref_logits = reference(**kwargs).float()
        with torch.no_grad():
            teacher_logits = teacher(**kwargs).float()
        p, q = ref_logits.log_softmax(-1), teacher_logits.log_softmax(-1)
        token_kl = (p.exp() * (p - q)).sum(-1)
        expected = reducer(torch.cat((token_kl[0, 2:5], token_kl[0, 7:9])))
        expected.backward()
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0.005)
        for (name, parameter), (ref_name, ref_parameter) in zip(
            student.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == ref_name
            torch.testing.assert_close(
                parameter.main_grad,
                ref_parameter.grad.float(),
                atol=0.012,
                rtol=0.05,
                msg=lambda message, name=name: f"{name}: {message}",
            )
        runtime.client.close()
    finally:
        if handle is not None:
            handle.stop()
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tied", [True, False])
def test_http_to_gpt_training_step(tmp_path, tied):
    mp.spawn(_step_worker, args=("file://" + str(tmp_path / "rendezvous"), tied), nprocs=1, join=True)
