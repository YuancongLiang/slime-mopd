"""SM80+ FP32 vocabulary statistics/dZ; imported only for the TileLang backend."""

from functools import lru_cache

import tilelang
import torch
import torch.distributed as dist
from tilelang import language as T

from .linear_kl import _reduce


@tilelang.jit(out_idx=[])
def _max_kernel(vocab, valid, block=1024):
    rows = T.dynamic("rows")
    tiles = T.ceildiv(vocab, block)

    @T.prim_func
    def kernel(
        S: T.Tensor((rows, vocab), "float32"),
        U: T.Tensor((rows, vocab), "float32"),
        Out: T.Tensor((2, rows, tiles), "float32"),
    ):
        with T.Kernel(rows, tiles, threads=128) as (row, tile):
            x = T.alloc_fragment((2, block), "float32")
            maximum = T.alloc_fragment((2,), "float32")
            for i, j in T.Parallel(2, block):
                col = tile * block + j
                x[i, j] = T.if_then_else(
                    col < valid, T.if_then_else(i == 0, S[row, col], U[row, col]), -T.infinity("float32")
                )
            T.reduce_max(x, maximum, dim=1)
            for i in T.Parallel(2):
                Out[i, row, tile] = maximum[i]

    return kernel


@tilelang.jit(out_idx=[])
def _moment_kernel(vocab, valid, block=1024):
    rows = T.dynamic("rows")
    tiles = T.ceildiv(vocab, block)

    @T.prim_func
    def kernel(
        S: T.Tensor((rows, vocab), "float32"),
        U: T.Tensor((rows, vocab), "float32"),
        Maximum: T.Tensor((2, rows), "float32"),
        Out: T.Tensor((3, rows, tiles), "float32"),
    ):
        with T.Kernel(rows, tiles, threads=128) as (row, tile):
            values = T.alloc_fragment((3, block), "float32")
            sums = T.alloc_fragment((3,), "float32")
            for i, j in T.Parallel(3, block):
                col = tile * block + j
                a = T.if_then_else(col < valid, S[row, col] - Maximum[0, row], 0.0)
                b = T.if_then_else(col < valid, U[row, col] - Maximum[1, row], 0.0)
                values[i, j] = T.if_then_else(
                    col < valid,
                    T.if_then_else(i == 0, T.exp(a), T.if_then_else(i == 1, T.exp(b), T.exp(a) * (a - b))),
                    0.0,
                )
            T.reduce_sum(values, sums, dim=1)
            for i in T.Parallel(3):
                Out[i, row, tile] = sums[i]

    return kernel


@tilelang.jit(out_idx=[])
def _grad_kernel(vocab, valid, temperature, block=1024):
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        S: T.Tensor((rows, vocab), "float32"),
        U: T.Tensor((rows, vocab), "float32"),
        Stats: T.Tensor((4, rows), "float32"),
        Upstream: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(rows, T.ceildiv(vocab, block), threads=128) as (row, tile):
            for j in T.Parallel(block):
                col = tile * block + j
                if col < vocab:
                    if col < valid:
                        a = S[row, col] - Stats[0, row]
                        b = U[row, col] - Stats[1, row]
                        S[row, col] = T.exp(a) / Stats[2, row] * (a - b - Stats[3, row]) * Upstream[row] / temperature
                    else:
                        S[row, col] = 0.0

    return kernel


@lru_cache(maxsize=32)
def _buffers(rows, vocab, device_index, stream):
    device = torch.device("cuda", device_index)
    tiles = (vocab + 1023) // 1024
    return [
        torch.empty(shape, dtype=torch.float32, device=device)
        for shape in ((2, rows, tiles), (2, rows), (3, rows, tiles), (3, rows), (4, rows), (rows,))
    ]


def _scratch(s):
    return _buffers(*s.shape, s.device.index, torch.cuda.current_stream(s.device).cuda_stream)


def statistics(s, t, valid, group):
    if valid == 0 and (group is None or dist.get_world_size(group) == 1):
        zeros = s.new_zeros(s.shape[0], dtype=torch.float32)
        return zeros, s.new_zeros((4, s.shape[0]), dtype=torch.float32)
    partial_max, maximum, partial_sum, moments, _, _ = _scratch(s)
    _max_kernel(s.shape[1], valid)(s, t, partial_max)
    torch.amax(partial_max, dim=-1, out=maximum)
    _reduce(maximum, dist.ReduceOp.MAX, group)
    _moment_kernel(s.shape[1], valid)(s, t, maximum, partial_sum)
    torch.sum(partial_sum, dim=-1, out=moments)
    _reduce(moments, dist.ReduceOp.SUM, group)
    mean = moments[2] / moments[0]
    return mean - moments[0].log() + moments[1].log(), torch.stack((maximum[0], maximum[1], moments[0], mean))


def gradient(s, t, valid, stats, upstream, temperature):
    *_, stats_buffer, upstream_buffer = _scratch(s)
    stats_buffer.copy_(stats)
    upstream_buffer.copy_(upstream)
    _grad_kernel(s.shape[1], valid, temperature)(s, t, stats_buffer, upstream_buffer)
    return s
