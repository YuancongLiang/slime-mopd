"""Benchmark OPD head+loss forward/backward on fixed hidden states, without backbones."""

import argparse
import json
import os
import tempfile

import torch
import torch.distributed as dist

from .linear_kl import Workspace, fused_linear_reverse_kl


def sampled_policy_loss(logits, labels, old_log_probs, teacher_log_probs, *, chunk_tokens=256):
    """Slime's zero-task-reward OPD advantages and clipped policy loss, including entropy logging."""
    from slime.utils.ppo_utils import calculate_log_probs_and_entropy, compute_policy_loss

    log_probs, entropy = calculate_log_probs_and_entropy(
        logits, labels, None, with_entropy=True, chunk_size=chunk_tokens, with_entropy_grad=False
    )
    log_probs = log_probs.flatten()
    old_log_probs, teacher_log_probs = old_log_probs.flatten(), teacher_log_probs.flatten()
    advantages = -(old_log_probs - teacher_log_probs)
    losses, _ = compute_policy_loss(old_log_probs - log_probs, advantages, 0.2, 0.2)
    return losses, entropy


def _measure(run, iterations):
    run()
    run()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        value = run()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iterations
    peak = torch.cuda.max_memory_allocated()
    return {
        "forward_backward_ms": ms,
        "peak_allocated_gib": peak / 1024**3,
        "incremental_peak_gib": (peak - baseline) / 1024**3,
        "mean_objective": value.float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--backend", choices=["torch", "tilelang"], default="tilelang")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--objective", choices=["full_vocab_reverse_kl", "sampled", "both"], default="full_vocab_reverse_kl"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.rows, args.hidden_size, args.vocab_size, args.chunk_tokens, args.iterations) <= 0:
        parser.error("All dimensions and iteration counts must be positive")
    rank, world = int(os.getenv("LOCAL_RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    if args.objective != "full_vocab_reverse_kl" and world != 1:
        parser.error("The sampled head microbenchmark currently supports TP=1 only; use replay for TP comparisons")
    torch.cuda.set_device(rank)
    rendezvous = tempfile.TemporaryDirectory()
    if world > 1:
        dist.init_process_group("nccl")
    elif args.objective != "full_vocab_reverse_kl":
        dist.init_process_group("nccl", init_method=f"file://{rendezvous.name}/store", rank=0, world_size=1)
    group = dist.group.WORLD if world > 1 else None
    try:
        torch.manual_seed(args.seed)
        local_vocab = (args.vocab_size + world - 1) // world
        h = torch.randn(args.rows, args.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        th = torch.randn_like(h)
        torch.manual_seed(args.seed + 1 + rank)
        w = (torch.randn(local_vocab, args.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_()
        tw = torch.randn_like(w) * 0.02
        w.main_grad = torch.zeros_like(w, dtype=torch.float32)
        w.grad_added_to_main_grad = False

        def reset_grad():
            h.grad = w.grad = None
            w.main_grad.zero_()

        result = vars(args) | {"gpu": torch.cuda.get_device_name(), "tp": world, "rank": rank}
        measured = {}
        if args.objective != "full_vocab_reverse_kl":
            from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce

            from slime.utils.ppo_utils import calculate_log_probs_and_entropy

            # Targets are prepared once and excluded from learner timing. This is not teacher-backbone timing.
            with torch.no_grad():
                logits = h @ w.t()
                labels = torch.multinomial(logits.float().softmax(-1), 1).squeeze(-1)
                old_log_probs, _ = calculate_log_probs_and_entropy(logits, labels, None)
                del logits
                target_start, target_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                target_start.record()
                teacher_log_probs, _ = calculate_log_probs_and_entropy(th @ tw.t(), labels, None)
                target_end.record()
                torch.cuda.synchronize()
                result["sampled_target_head_logprob_setup_ms"] = target_start.elapsed_time(target_end)

            def run_sampled():
                reset_grad()
                logits = linear_with_grad_accumulation_and_async_allreduce(
                    h, w, None, True, False, False, tp_group=dist.group.WORLD
                )
                value, _ = sampled_policy_loss(
                    logits, labels, old_log_probs, teacher_log_probs, chunk_tokens=args.chunk_tokens
                )
                value.mean().backward()
                return value

            measured["sampled"] = _measure(run_sampled, args.iterations)
            result["sampled_reverse_kl_estimate"] = (old_log_probs - teacher_log_probs).mean().item()
            del labels, old_log_probs, teacher_log_probs
            reset_grad()
            torch.cuda.empty_cache()

        workspace = Workspace()

        def run_full():
            reset_grad()
            value = fused_linear_reverse_kl(
                h,
                w,
                th,
                tw,
                tp_group=group,
                vocab_size=args.vocab_size,
                backend=args.backend,
                chunk_tokens=args.chunk_tokens,
                workspace=workspace,
                accumulate_main_grad=True,
            )
            value.mean().backward()
            return value

        if args.objective != "sampled":
            measured["full_vocab_reverse_kl"] = _measure(run_full, args.iterations)
        for value in measured.values():
            value["response_tokens_per_second"] = args.rows / (value["forward_backward_ms"] / 1000)
        if args.objective == "both":
            ratio = (
                measured["full_vocab_reverse_kl"]["forward_backward_ms"] / measured["sampled"]["forward_backward_ms"]
            )
            result |= {
                "results": measured,
                "head_loss_time_multiplier": ratio,
                "head_loss_overhead_percent": (ratio - 1) * 100,
                "head_loss_throughput_drop_percent": (1 - 1 / ratio) * 100,
            }
        else:
            result |= measured[args.objective]
        if "full_vocab_reverse_kl" in measured:
            result["mean_kl"] = measured["full_vocab_reverse_kl"]["mean_objective"]
        print(json.dumps(result, indent=2))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        rendezvous.cleanup()


if __name__ == "__main__":
    main()
