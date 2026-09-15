from argparse import Namespace
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from slime.observability.opd_metrics import Collector
from slime.observability.opd_timing import backward_timing, finish_loss, sampled_head, sampled_loss_forward


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(8, 8)
        self.output_layer = nn.Linear(8, 16)

    def forward(self, inputs):
        return self.output_layer(self.backbone(inputs).tanh())


@pytest.fixture
def collector(monkeypatch):
    value = Collector()
    monkeypatch.setattr("slime.observability.opd_timing.get_collector", lambda: value)
    return value


def _args(**kwargs):
    return Namespace(
        use_opd=True, opd_objective="sampled", pipeline_model_parallel_size=1, context_parallel_size=1, **kwargs
    )


def test_disabled_timing_preserves_graph_and_config(collector):
    model, batch, config = Model(), {}, Namespace(timers=None)
    with backward_timing(config), sampled_head(_args(), model, batch):
        output = model(torch.randn(4, 8))
        with sampled_loss_forward(batch):
            loss = output.square().mean()
        assert finish_loss(batch, loss) is loss
        loss.backward()
        assert config.timers is None
    assert batch == {}
    assert collector.events == []
    assert "forward" not in model.output_layer.__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU event timing")
@pytest.mark.parametrize("recompute", [False, True])
def test_sampled_gpu_timing_preserves_gradients_and_restores_hooks(collector, recompute):
    collector.enabled = collector.profile_gpu = True
    model = Model().cuda()
    reference = deepcopy(model)
    inputs = torch.randn(4, 8, device="cuda", requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    batch, config = {}, Namespace(timers=None)

    with backward_timing(config), sampled_head(_args(), model, batch):
        output = model(inputs)

        def loss_function(logits):
            return logits.log_softmax(-1).square().mean()

        with sampled_loss_forward(batch):
            loss = checkpoint(loss_function, output, use_reentrant=False) if recompute else loss_function(output)
        loss = finish_loss(batch, 2.5 * loss)
        config.timers("forward-compute").start().stop()
        config.timers("backward-compute").start()
        loss.backward()
        config.timers("backward-compute").stop()

    (2.5 * loss_function(reference(reference_inputs))).backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, expected.grad)
    assert config.timers is None
    assert "forward" not in model.output_layer.__dict__
    assert batch == {}
    counters = collector.snapshot()["counters"]
    assert set(counters) == {"head_loss_forward_s", "head_loss_backward_s", "learner_backward_s"}
    assert all(value > 0 for value in counters.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU event timing")
def test_timing_restores_existing_instance_forward_and_timers_on_error(collector):
    collector.enabled = collector.profile_gpu = True
    model = Model().cuda()
    original = model.output_layer.forward
    model.output_layer.forward = original

    def previous_timers(name, **kwargs):
        return None

    config = Namespace(timers=previous_timers)
    with pytest.raises(RuntimeError, match="test error"):
        with backward_timing(config), sampled_head(_args(), model, {}):
            raise RuntimeError("test error")
    assert config.timers is previous_timers
    assert model.output_layer.forward is original
