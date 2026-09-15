"""Sampled OPD head/loss timing without changing Megatron's global functions."""

from contextlib import contextmanager

import torch

from .opd_metrics import get_collector


class _BackwardBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, state, beginning):
        ctx.state, ctx.beginning = state, beginning
        # MCore scales the scalar loss in place after calling its loss closure.
        return tensor.clone() if beginning else tensor

    @staticmethod
    def backward(ctx, grad):
        collector, start = ctx.state
        if ctx.beginning:
            start.record()
        else:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            collector.events.append(("head_loss_backward_s", start, end))
        return grad, None, None


@contextmanager
def sampled_head(args, model, batch):
    collector = get_collector()
    enabled = (
        collector.profile_gpu
        and args.use_opd
        and getattr(args, "opd_objective", "sampled") == "sampled"
        and args.pipeline_model_parallel_size == args.context_parallel_size == 1
    )
    if not enabled:
        yield
        return

    from slime.opd.megatron import head_forward, unwrap

    original = unwrap(model).output_layer.forward
    state = (collector, torch.cuda.Event(enable_timing=True))
    batch["_opd_head_timing"] = state

    def forward(layer, hidden, *positional, **kwargs):
        hidden = _BackwardBoundary.apply(hidden, state, False)
        with collector.time("head_loss_forward", gpu=True):
            return original(hidden, *positional, **kwargs)

    with head_forward(model, forward):
        yield


@contextmanager
def sampled_loss_forward(batch):
    if "_opd_head_timing" not in batch:
        yield
        return
    with get_collector().time("head_loss_forward", gpu=True):
        yield


def finish_loss(batch, loss):
    state = batch.pop("_opd_head_timing", None)
    return _BackwardBoundary.apply(loss, state, True) if state is not None else loss


class _BackwardTimer:
    def __init__(self, collector, wrapped=None):
        self.collector = collector
        self.wrapped = wrapped

    def start(self, *args, **kwargs):
        if self.wrapped is not None:
            self.wrapped.start(*args, **kwargs)
        self.event = torch.cuda.Event(enable_timing=True)
        self.event.record()
        return self

    def stop(self, *args, **kwargs):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.collector.events.append(("learner_backward_s", self.event, end))
        if self.wrapped is not None:
            self.wrapped.stop(*args, **kwargs)


class _NoTimer:
    def start(self, *args, **kwargs):
        return self

    def stop(self, *args, **kwargs):
        pass


@contextmanager
def backward_timing(config):
    """Use MCore's schedule timer hook to include recomputation and DDP launches."""
    collector = get_collector()
    if not collector.profile_gpu:
        yield
        return

    previous = config.timers
    previous_finalize = getattr(config, "finalize_model_grads_func", None)
    timer = _BackwardTimer(collector, previous("backward-compute", log_level=2) if previous else None)
    noop = _NoTimer()

    def timers(name, **kwargs):
        if name == "backward-compute":
            return timer
        return previous(name, **kwargs) if previous else noop

    config.timers = timers
    if previous_finalize is not None:

        def finalize(*args, **kwargs):
            with collector.time("learner_finalize", gpu=True):
                return previous_finalize(*args, **kwargs)

        config.finalize_model_grads_func = finalize
    try:
        yield
    finally:
        config.timers = previous
        if previous_finalize is not None:
            config.finalize_model_grads_func = previous_finalize
