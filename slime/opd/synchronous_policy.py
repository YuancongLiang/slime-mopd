"""Coordinator-owned student version binding for a synchronous rollout batch.

No data-row or deployment version is accepted as evidence. The coordinator supplies
its published version; RolloutManager checks every managed engine before and after
executing the batch. The single coordinator must own all weight publications.
"""
from dataclasses import dataclass
from urllib.parse import urlsplit
from uuid import uuid4


def validate_sync_config(args):
    cfg = getattr(args, "opd_resolved", {})
    if not (getattr(args, "use_opd", False) and getattr(args, "opd_objective", None) == "full_vocab_reverse_kl"):
        raise ValueError("Synchronous policy binding requires full-vocabulary MOPD")
    if cfg.get("sync_policy_binding") is not True or cfg.get("prefetch_rollouts") is not False:
        raise ValueError("Synchronous policy binding requires sync_policy_binding=true and prefetch_rollouts=false")
    for option in ("use_fault_tolerance", "rollout_external", "debug_train_only", "debug_rollout_only",
                   "load_debug_rollout_data", "offload_train", "offload_rollout", "release_train"):
        if getattr(args, option, False):
            raise ValueError(f"Synchronous policy binding does not support {option}")
    if getattr(args, "update_weights_interval", 1) != 1:
        raise ValueError("Synchronous policy binding requires update_weights_interval=1")


def engine_endpoint(url):
    parts = urlsplit(url)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment or parts.path.rstrip("/") not in {"", "/v1"}):
        raise ValueError("Managed policy endpoint must be an engine root or /v1 URL")
    return parts._replace(path="/v1").geturl()


@dataclass
class SyncPolicyBinding:
    batch_id: int
    version: str
    endpoints: tuple[str, ...]
    binding_id: str
    active: bool = True

    def endpoint(self, requested, index=0):
        if not self.active:
            raise ValueError("Synchronous policy binding has expired")
        if requested == "managed":
            return self.endpoints[(index or 0) % len(self.endpoints)]
        endpoint = engine_endpoint(requested)
        if endpoint not in self.endpoints:
            raise ValueError("Harbor student endpoint is not a verified managed engine; routers are not supported")
        return endpoint


def require_binding(args):
    validate_sync_config(args)
    binding = getattr(args, "_opd_sync_policy_binding", None)
    if not isinstance(binding, SyncPolicyBinding) or not binding.active:
        raise ValueError("Harbor requires an active coordinator binding with actual_weight_version")
    return binding


def published_version(actor):
    """Read the learner's publication clock, not a rollout counter."""
    version = str(actor.weight_updater.weight_version)
    if version not in actor.args._opd_version_steps:
        raise ValueError("Student policy version has not been published by this learner")
    return version


def run_bound_batch(manager, rollout_id, expected_version, *, evaluation=False):
    import ray

    args = manager.args
    validate_sync_config(args)
    if getattr(args, "_opd_sync_policy_binding", None) is not None:
        raise ValueError("Overlapping synchronous policy batches are not supported")
    if not isinstance(expected_version, str) or not expected_version:
        raise ValueError("Confirmed learner policy version required")
    servers = [s for s in manager.servers.values() if s.update_weights]
    if len(servers) != 1:
        raise ValueError("Synchronous policy binding requires exactly one updatable model")
    server = servers[0]
    engines = tuple(server.engines)
    if not engines or any(e is None for e in engines):
        raise ValueError("All managed student engines must be available")

    def snapshot():
        if tuple(server.engines) != engines or server.num_new_engines:
            raise ValueError("Student engine membership changed or weights have not been published")
        urls = ray.get([e.get_url.remote() for e in engines])
        versions = ray.get([e.get_weight_version.remote() for e in engines])
        if any(v is None or str(v) != expected_version for v in versions):
            raise ValueError("Managed engine version differs from the learner's published policy")
        endpoints = tuple(engine_endpoint(url) for url in urls)
        if len(set(endpoints)) != len(engines):
            raise ValueError("Managed engines advertise duplicate endpoints")
        return endpoints

    endpoints = snapshot()
    binding = SyncPolicyBinding(rollout_id, expected_version, endpoints, uuid4().hex)
    args._opd_sync_policy_binding = binding
    try:
        result = manager.eval(rollout_id) if evaluation else manager.generate(rollout_id)
        if snapshot() != endpoints:
            raise ValueError("Managed student endpoints changed during rollout")
        return result
    finally:
        binding.active = False
        del args._opd_sync_policy_binding
