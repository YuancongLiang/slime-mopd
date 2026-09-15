"""Full-vocabulary KL value, gradient, masking and workspace regression tests."""

import pytest
import torch

from slime.opd.linear_kl import Workspace, fused_linear_reverse_kl

NUM_GPUS = 0


@pytest.mark.parametrize("chunk", [1, 4, 32])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_dense_loss_and_gradients(chunk, temperature):
    torch.manual_seed(19)
    h = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
    w = torch.randn(17, 5, dtype=torch.float64, requires_grad=True)
    th, tw = torch.randn_like(h), torch.randn_like(w)
    upstream = torch.randn(9, dtype=torch.float64)
    p = (h @ w[:13].t() / temperature).log_softmax(-1)
    q = (th @ tw[:13].t() / 1.3).log_softmax(-1)
    expected = (p.exp() * (p - q)).sum(-1)
    ref_grad = torch.autograd.grad((expected * upstream).sum(), (h, w))
    loss = fused_linear_reverse_kl(
        h, w, th, tw, vocab_size=13, chunk_tokens=chunk, student_temperature=temperature, teacher_temperature=1.3
    )
    grads = torch.autograd.grad((loss * upstream).sum(), (h, w))
    torch.testing.assert_close(loss, expected)
    for actual, ref in zip(grads, ref_grad, strict=True):
        torch.testing.assert_close(actual, ref)
    assert torch.count_nonzero(grads[1][13:]) == 0


def test_inflight_graphs_reuse_scratch_without_corruption():
    torch.manual_seed(8)
    w = torch.randn(11, 3, dtype=torch.float64, requires_grad=True)
    h = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    workspace = Workspace()
    teacher = torch.randn_like(w)
    a = fused_linear_reverse_kl(h, w, h.detach(), teacher, chunk_tokens=3, workspace=workspace)
    b = fused_linear_reverse_kl(h, w, h.detach(), teacher * 2, chunk_tokens=3, workspace=workspace)
    (a.sum() + b.sum()).backward()
    actual = w.grad.clone()
    w.grad = None
    p = (h @ w.t()).log_softmax(-1)
    q1 = (h.detach() @ teacher.t()).log_softmax(-1)
    q2 = (h.detach() @ (teacher * 2).t()).log_softmax(-1)
    (p.exp() * (2 * p - q1 - q2)).sum().backward()
    torch.testing.assert_close(w.grad, actual)
    assert len(workspace.buffers) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vocab,valid", [(2053, 2001), (4096, 4096), (512, 0)])
def test_tilelang_statistics_and_gradient(vocab, valid):
    from slime.opd.linear_kl import _gradient, _stats

    if valid == 0:
        s = torch.randn(5, vocab, device="cuda")
        t = torch.randn_like(s)
        actual, stats = _stats(s, t, valid, None, "torch")
        assert torch.isfinite(actual).all() and torch.isfinite(stats).all()
        return
    torch.manual_seed(7)
    s = torch.randn(5, vocab, device="cuda")
    t = torch.randn_like(s)
    ref, stats = _stats(s.clone(), t.clone(), valid, None, "torch")
    actual, actual_stats = _stats(s.clone(), t.clone(), valid, None, "tilelang")
    torch.testing.assert_close(actual, ref, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_stats, stats, atol=2e-5, rtol=2e-5)
    upstream = torch.randn(5, device="cuda")
    ref_grad = _gradient(s.clone(), t.clone(), valid, stats, upstream, 0.7, "torch")
    grad = _gradient(s.clone(), t.clone(), valid, actual_stats, upstream, 0.7, "tilelang")
    torch.testing.assert_close(grad, ref_grad, atol=1e-7, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tilelang_linear_backward():
    torch.manual_seed(18)
    h = torch.randn(19, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(1027, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    th, tw = torch.randn_like(h), torch.randn_like(w)
    ref = fused_linear_reverse_kl(h, w, th, tw, vocab_size=1001, chunk_tokens=8)
    grad = torch.autograd.grad(ref.sum(), (h, w))
    actual = fused_linear_reverse_kl(h, w, th, tw, vocab_size=1001, chunk_tokens=8, backend="tilelang")
    actual_grad = torch.autograd.grad(actual.sum(), (h, w))
    torch.testing.assert_close(actual, ref, atol=1e-4, rtol=2e-5)
    for a, b in zip(actual_grad, grad, strict=True):
        torch.testing.assert_close(a, b, atol=0.03125, rtol=0.02)
