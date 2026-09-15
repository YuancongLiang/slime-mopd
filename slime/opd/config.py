"""Strict configuration for the opt-in MOPD path; no GPU imports."""

import hashlib
import json
import logging
import math

import yaml


def enabled(args):
    return getattr(args, "use_opd", False) and getattr(args, "opd_objective", "sampled") == "full_vocab_reverse_kl"


def _keys(value, allowed, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {where} keys: {sorted(unknown)}")


def _positive(value, name, integer=False, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (not zero and value == 0) or (integer and not isinstance(value, int)):
        raise ValueError(f"Invalid {name}: {value}")


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    _keys(raw, {"schema_version", "opd"}, "root")
    if raw.get("schema_version") != 1:
        raise ValueError("OPD schema_version must be 1")
    cfg = raw.get("opd")
    defaults = {
        "backend": "tilelang",
        "chunk_tokens": 256,
        "student_temperature": 1.0,
        "teacher_temperature": 1.0,
        "coefficient": 1.0,
        "domain_key": "domain",
        "head_cache_bytes": 8 * 1024**3,
        "timeout_seconds": 1800,
        "max_context_tokens": 32768,
        "prefetch_rollouts": False,
        "max_policy_lag_optimizer_steps": 1,
    }
    _keys(cfg, {*defaults, "teachers", "domains", "placement"}, "opd")
    cfg = defaults | cfg
    if cfg["backend"] not in {"torch", "tilelang"}:
        raise ValueError("OPD backend must be torch or tilelang")
    for key in ("chunk_tokens", "head_cache_bytes", "max_context_tokens"):
        _positive(cfg[key], key, integer=True)
    _positive(cfg["max_policy_lag_optimizer_steps"], "max_policy_lag_optimizer_steps", integer=True, zero=True)
    if type(cfg["prefetch_rollouts"]) is not bool:
        raise ValueError("prefetch_rollouts must be a boolean")
    if cfg["prefetch_rollouts"] and cfg["max_policy_lag_optimizer_steps"] == 0:
        raise ValueError("Prefetched rollouts require at least one optimizer step of allowed policy lag")
    for key in ("student_temperature", "teacher_temperature", "timeout_seconds"):
        _positive(cfg[key], key)
    _positive(cfg["coefficient"], "coefficient", zero=True)
    if not isinstance(cfg["domain_key"], str) or not cfg["domain_key"]:
        raise ValueError("domain_key must be a nonempty string")
    for key in ("teachers", "domains"):
        if not isinstance(cfg.get(key), dict) or not cfg[key]:
            raise ValueError(f"{key} must be a nonempty mapping")
    for name, teacher in cfg["teachers"].items():
        _keys(teacher, {"version", "endpoints", "model_hash"}, f"teacher {name}")
        if "model_hash" in teacher:
            value = teacher["model_hash"]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Teacher {name} model_hash must be a lowercase SHA256 digest")
        if not isinstance(name, str) or not isinstance(teacher.get("version"), str) or not teacher["version"]:
            raise ValueError("Teacher id and immutable version must be strings")
        urls = teacher.get("endpoints")
        if (
            not isinstance(urls, list)
            or not urls
            or any(not isinstance(url, str) or not url.startswith(("http://", "https://")) for url in urls)
        ):
            raise ValueError(f"Teacher {name} requires HTTP endpoints")
    for name, domain in cfg["domains"].items():
        _keys(domain, {"teacher", "sampling_weight"}, f"domain {name}")
        if not isinstance(name, str) or domain.get("teacher") not in cfg["teachers"]:
            raise ValueError(f"Invalid teacher for domain {name}")
        _positive(domain.get("sampling_weight"), f"{name}.sampling_weight", zero=True)
    if sum(d["sampling_weight"] for d in cfg["domains"].values()) <= 0:
        raise ValueError("Domain sampling weights must have a positive sum")
    if "placement" in cfg:
        _keys(cfg["placement"], {"learner_nodes", "rollout_nodes"}, "placement")
        for role in ("learner_nodes", "rollout_nodes"):
            nodes = cfg["placement"].get(role)
            if not isinstance(nodes, list) or not nodes or any(not isinstance(n, str) for n in nodes):
                raise ValueError(f"placement.{role} must list Ray node IP addresses")
        if set(cfg["placement"]["learner_nodes"]) & set(cfg["placement"]["rollout_nodes"]):
            raise ValueError("MOPD learner and rollout nodes must be disjoint")
    cfg["fingerprint"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    semantics = {
        "domains": cfg["domains"],
        "domain_key": cfg["domain_key"],
        "teachers": {name: item["version"] for name, item in cfg["teachers"].items()},
    }
    model_hashes = {name: item["model_hash"] for name, item in cfg["teachers"].items() if "model_hash" in item}
    if model_hashes:
        semantics["teacher_model_hashes"] = model_hashes
    cfg["sampling_fingerprint"] = hashlib.sha256(json.dumps(semantics, sort_keys=True).encode()).hexdigest()
    return cfg


def configure(args):
    if not enabled(args):
        if getattr(args, "opd_objective", "sampled") == "full_vocab_reverse_kl":
            raise ValueError("--opd-objective full_vocab_reverse_kl requires --use-opd")
        if getattr(args, "opd_config", None):
            raise ValueError("--opd-config requires --use-opd --opd-objective full_vocab_reverse_kl")
        return
    if not getattr(args, "opd_config", None):
        raise ValueError("Full-vocabulary OPD requires --opd-config")
    cfg = load_config(args.opd_config)
    if getattr(args, "opd_type", None) is not None or getattr(args, "opd_teacher_load", None) is not None:
        raise ValueError("Full-vocabulary OPD uses the teacher registry, not --opd-type/--opd-teacher-load")
    if getattr(args, "loss_type", "policy_loss") not in {"policy_loss", "full_vocab_opd"}:
        raise ValueError("Full-vocabulary OPD cannot be combined with another loss type")
    for name in (
        "use_critic",
        "use_kl_loss",
        "keep_old_actor",
        "enable_mtp_training",
        "use_routing_replay",
        "use_rollout_routing_replay",
        "allgather_cp",
        "calculate_per_token_loss",
        "colocate",
        "recompute_loss_function",
        "tp_comm_overlap",
        "use_megatron_fsdp",
        "use_torch_fsdp2",
        "partial_rollout",
    ):
        if getattr(args, name, False):
            raise ValueError(f"Full-vocabulary OPD does not yet support {name}")
    for name in ("pipeline_model_parallel_size", "context_parallel_size"):
        if getattr(args, name, 1) != 1:
            raise ValueError(f"Full-vocabulary OPD requires {name}=1")
    if getattr(args, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        raise ValueError("Full-vocabulary OPD does not support VPP")
    if getattr(args, "kl_coef", 0) != 0:
        raise ValueError("Full-vocabulary OPD requires --kl-coef 0")
    if not getattr(args, "bf16", False):
        raise ValueError("Full-vocabulary OPD currently requires --bf16")
    if getattr(args, "fp8", None):
        raise ValueError("Full-vocabulary OPD does not support FP8")
    if getattr(args, "custom_reward_post_process_path", None) is not None:
        raise ValueError("Pure MOPD cannot use a sampled-OPD reward postprocessor")
    if not getattr(args, "rollout_global_dataset", False):
        raise ValueError(
            "Full-vocabulary MOPD requires the global rollout dataset; remove --disable-rollout-global-dataset"
        )
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("Full-vocabulary MOPD currently requires the Megatron training backend")
    if getattr(args, "opd_resolved", None) != cfg:
        logging.getLogger(__name__).info(
            "Resolved full-vocabulary OPD configuration: %s", json.dumps(cfg, sort_keys=True)
        )
    args.opd_resolved = cfg
    args.loss_type = "full_vocab_opd"
    args.compute_advantages_and_returns = False
    args.rewards_normalization = False
    if getattr(args, "custom_rm_path", None) not in (None, "slime.opd.rollout.reward_func"):
        raise ValueError("Pure MOPD requires its hidden-target reward hook")
    args.custom_rm_path = "slime.opd.rollout.reward_func"
    return cfg
