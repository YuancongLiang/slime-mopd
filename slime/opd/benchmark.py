"""Benchmark the complete linear+KL forward/backward after compilation warmup."""

import argparse
import json
import os

import torch
import torch.distributed as dist

from .linear_kl import Workspace, fused_linear_reverse_kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--backend", choices=["torch", "tilelang"], default="tilelang")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if min(args.rows, args.hidden_size, args.vocab_size, args.chunk_tokens, args.iterations) <= 0:
        parser.error("All dimensions and iteration counts must be positive")
    rank, world = int(os.getenv("LOCAL_RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl")
    group = dist.group.WORLD if world > 1 else None
    try:
        torch.manual_seed(42)
        local_vocab = (args.vocab_size + world - 1) // world
        h = torch.randn(args.rows, args.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        th = torch.randn_like(h)
        w = (torch.randn(local_vocab, args.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_()
        tw = torch.randn_like(w) * 0.02
        w.main_grad = torch.zeros_like(w, dtype=torch.float32)
        w.grad_added_to_main_grad = False
        workspace = Workspace()

        def run():
            h.grad = w.grad = None
            w.main_grad.zero_()
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

        run()
        run()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            loss = run()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / args.iterations
        peak = torch.cuda.max_memory_allocated()
        print(
            json.dumps(
                vars(args)
                | {
                    "gpu": torch.cuda.get_device_name(),
                    "tp": world,
                    "rank": rank,
                    "forward_backward_ms": ms,
                    "response_tokens_per_second": args.rows / (ms / 1000),
                    "peak_allocated_gib": peak / 1024**3,
                    "incremental_peak_gib": (peak - baseline) / 1024**3,
                    "mean_kl": loss.float().mean().item(),
                },
                indent=2,
            )
        )
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
