"""Real MCore schedule coverage for sampled OPD metrics and profiling hooks."""

from argparse import Namespace
from collections import Counter
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

NUM_GPUS = 1


def _schedule_worker(rank, rendezvous):
    torch.cuda.set_device(3 if torch.cuda.device_count() > 3 else 0)
    from megatron.core import mpu, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_config import TransformerConfig

    from slime.backends.megatron_utils.data import DataIterator
    from slime.backends.megatron_utils.model import train_one_step
    from slime.observability.opd_metrics import get_collector

    dist.init_process_group("nccl", init_method=rendezvous, rank=0, world_size=1, timeout=timedelta(seconds=120))
    mpu.initialize_model_parallel()
    tensor_parallel.model_parallel_cuda_manual_seed(123)
    collector = get_collector()
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
        student = (
            GPTModel(
                config,
                get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=128,
                max_sequence_length=64,
                share_embeddings_and_output_weights=True,
                position_embedding_type="rope",
            )
            .cuda()
            .bfloat16()
        )
        wrapped = DistributedDataParallel(
            config, DistributedDataParallelConfig(grad_reduce_in_fp32=True), Float16Module(config, student)
        )
        args = Namespace(
            use_opd=True,
            opd_objective="sampled",
            opd_metrics_interval=1,
            custom_megatron_before_train_step_hook_path=None,
            save_debug_train_data=None,
            data_pad_size_multiplier=16,
            allgather_cp=False,
            enable_mtp_training=False,
            seq_length=16,
            micro_batch_size=1,
            decoder_seq_length=None,
            ci_test=False,
            check_for_nan_in_loss_and_grad=True,
            calculate_per_token_loss=False,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            rollout_top_p=1.0,
            rollout_temperature=1.0,
            log_probs_chunk_size=16,
            use_rollout_logprobs=False,
            use_opsm=False,
            advantage_estimator="grpo",
            eps_clip=0.2,
            eps_clip_high=0.2,
            eps_clip_c=None,
            get_mismatch_metrics=False,
            use_tis=False,
            entropy_coef=0.01,
            use_kl_loss=False,
            loss_type="policy_loss",
            recompute_loss_function=False,
        )
        data = {
            "tokens": [torch.tensor([1, 2, 3, 4, 5, 6], device="cuda"), torch.tensor([7, 8, 9, 10], device="cuda")],
            "total_lengths": [6, 4],
            "response_lengths": [3, 2],
            "loss_masks": [torch.tensor([1, 1, 0], device="cuda"), torch.tensor([1, 1], device="cuda")],
            "rollout_mask_sums": [torch.tensor(2, device="cuda"), torch.tensor(2, device="cuda")],
            "opd_reverse_kl": [torch.tensor([0.2, 0.4, 99], device="cuda"), torch.tensor([0.6, 0.8], device="cuda")],
            "opd_domains": ["math", "code"],
        }
        data["advantages"] = [-values for values in data["opd_reverse_kl"]]

        class Optimizer:
            def zero_grad(self):
                for parameter in student.parameters():
                    parameter.grad = None

            def step(self):
                wrapped.finish_grad_sync()
                self.gradients = [parameter.main_grad.detach().clone() for parameter in student.parameters()]
                return True, 0.0, 0

        optimizer = Optimizer()
        scheduler_steps = []
        scheduler = Namespace(step=lambda increment: scheduler_steps.append(increment))

        def train(profile):
            collector.__init__()
            if profile:
                collector.start(args, 0)
            with patch("slime.backends.megatron_utils.model.get_args", return_value=args):
                metrics, _ = train_one_step(
                    args, 0, 0, [DataIterator(data, [[0], [1]])], [wrapped], optimizer, scheduler, 2, 2
                )
            return metrics, optimizer.gradients

        baseline_metrics, baseline_gradients = train(False)
        metrics, gradients = train(True)
        assert metrics == pytest.approx(baseline_metrics)
        assert metrics["opd_reverse_kl"] == pytest.approx(0.5)
        for actual, expected in zip(gradients, baseline_gradients, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert scheduler_steps == [2, 2]
        event_counts = Counter(name for name, *_ in collector.events)
        assert event_counts["head_loss_forward_s"] == 4
        assert event_counts["head_loss_backward_s"] == event_counts["learner_backward_s"] == 2
        counters = collector.snapshot()["counters"]
        assert counters["effective_tokens"] == 4
        assert counters["computed_tokens"] == 5
        assert counters["updates"] == 1
        assert counters["skipped_updates"] == 0
        assert counters["domain/math/sampled_kl_token_sum"] == pytest.approx(0.6)
        assert counters["domain/math/sampled_kl_token_count"] == 2
        assert counters["domain/code/sampled_kl_token_sum"] == pytest.approx(1.4)
        assert counters["domain/code/sampled_kl_token_count"] == 2
        assert all(counters[name] > 0 for name in event_counts)
        assert config.timers is None
        assert "forward" not in student.output_layer.__dict__
    finally:
        collector.__init__()
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sampled_opd_metrics_in_actual_megatron_schedule(tmp_path):
    mp.spawn(_schedule_worker, args=("file://" + str(tmp_path / "rendezvous"),), nprocs=1, join=True)
