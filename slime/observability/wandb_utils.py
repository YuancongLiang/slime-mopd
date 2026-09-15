import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import wandb

logger = logging.getLogger(__name__)


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    mode = args.wandb_mode or os.environ.get("WANDB_MODE", "online")

    if mode == "online" and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    identity = _load_opd_run(args)
    run_id = args.wandb_run_id or identity.get("id")
    if identity and run_id == identity["id"]:
        group, run_name = identity["group"], identity["name"]
    elif args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "id": run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }
    if run_id and mode == "online":
        init_kwargs["resume"] = "allow"

    # Configure settings based on offline/online mode
    if mode == "online":
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)
    else:
        init_kwargs["settings"] = wandb.Settings(mode=mode)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id
    if getattr(args, "use_opd", False) and getattr(args, "save", None) and mode != "disabled":
        path = Path(args.save) / "wandb_run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"id": wandb.run.id, "group": group, "name": run_name}) + "\n")
        temporary.replace(path)


def _load_opd_run(args):
    if not getattr(args, "use_opd", False) or getattr(args, "finetune", False) or not getattr(args, "load", None):
        return {}
    directory = Path(args.load)
    path = directory / "wandb_run.json"
    if not (directory / "latest_checkpointed_iteration.txt").exists() or not path.exists():
        return {}
    return json.loads(path.read_text())


def _compute_config_for_logging(args):
    output = _args_to_config_dict(args)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    if getattr(args, "use_critic", False):
        critic_args = _get_role_args_for_logging(args, role="critic")
        output.update(_prefix_config_keys(_args_to_config_dict(critic_args), "critic"))

    return output


def _args_to_config_dict(args):
    return _redact_secrets(deepcopy(args.__dict__))


def _redact_secrets(value):
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            name = str(key).lower().replace("-", "_")
            sensitive = (
                name in {"wandb_key", "token", "authorization"}
                or name.endswith("_token")
                or any(
                    word in name
                    for word in ("password", "secret", "credential", "api_key", "access_key", "private_key")
                )
            )
            output[key] = "[REDACTED]" if sensitive and item is not None else _redact_secrets(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_redact_secrets(item) for item in value]
    return value


def _prefix_config_keys(config, prefix):
    return {f"{prefix}/{key}": value for key, value in config.items()}


def _get_role_args_for_logging(args, role):
    if getattr(args, "megatron_config_path", None) is None:
        return args

    from slime.utils.arguments import parse_megatron_role_args

    return parse_megatron_role_args(args, args.megatron_config_path, role=role)


def _compute_secondary_config_for_logging(args, role=None):
    config = _args_to_config_dict(args)
    if role == "critic":
        return _prefix_config_keys(config, "critic")
    return config


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, role=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    mode = args.wandb_mode or os.environ.get("WANDB_MODE", "online")

    if mode == "online" and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if mode == "online":
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )
    else:
        settings_kwargs = dict(mode=mode)

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_secondary_config_for_logging(args, role=role),
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("opd/step")
    wandb.define_metric("opd/*", step_metric="opd/step")
    wandb.define_metric("perf/opd_*", step_metric="opd/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
