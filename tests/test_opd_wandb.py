import json
import os
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from slime.observability import wandb_utils

NUM_GPUS = 0


def _args(**overrides):
    values = dict(
        use_wandb=True,
        use_opd=True,
        wandb_mode="offline",
        wandb_key=None,
        wandb_host=None,
        wandb_random_suffix=False,
        wandb_group="two-teacher",
        wandb_project="slime-mopd",
        wandb_team=None,
        wandb_run_id=None,
        wandb_dir=None,
        rank=0,
        save=None,
        load=None,
        finetune=False,
    )
    return Namespace(**(values | overrides))


@pytest.fixture
def sdk(monkeypatch):
    fake = SimpleNamespace(
        init=Mock(),
        login=Mock(),
        Settings=lambda **values: values,
        define_metric=Mock(),
        util=SimpleNamespace(generate_id=lambda: "suffix"),
    )

    def init(**kwargs):
        fake.run = SimpleNamespace(id=kwargs.get("id") or "generated-id")

    fake.init.side_effect = init
    monkeypatch.setattr(wandb_utils, "wandb", fake)
    return fake


def test_opd_resume_restores_identity_but_finetune_creates_new_run(tmp_path, sdk):
    saved = tmp_path / "saved"
    original = _args(save=str(saved), wandb_random_suffix=True)
    wandb_utils.init_wandb_primary(original)
    identity = json.loads((saved / "wandb_run.json").read_text())
    assert identity == {"id": "generated-id", "group": "two-teacher_suffix", "name": "two-teacher_suffix-RANK_0"}
    (saved / "latest_checkpointed_iteration.txt").write_text("8")

    resumed = _args(load=str(saved), save=str(tmp_path / "resumed"), wandb_mode="online")
    wandb_utils.init_wandb_primary(resumed)
    init = sdk.init.call_args.kwargs
    assert init["id"] == "generated-id"
    assert init["resume"] == "allow"
    assert init["name"] == identity["name"]
    assert init["settings"] == {"mode": "shared", "x_primary": True}

    wandb_utils.init_wandb_primary(_args(load=str(saved), finetune=True))
    assert sdk.init.call_args.kwargs["id"] is None
    assert sdk.init.call_args.kwargs["name"] == "two-teacher"


def test_initial_weights_and_generic_rl_do_not_automatically_resume(tmp_path, sdk):
    (tmp_path / "wandb_run.json").write_text(json.dumps({"id": "old", "group": "old", "name": "old"}))
    wandb_utils.init_wandb_primary(_args(load=str(tmp_path)))
    assert sdk.init.call_args.kwargs["id"] is None
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("8")
    wandb_utils.init_wandb_primary(_args(load=str(tmp_path), use_opd=False))
    assert sdk.init.call_args.kwargs["id"] is None


def test_explicit_run_id_and_offline_secondary_keep_same_id(tmp_path, sdk):
    args = _args(wandb_run_id="explicit", save=str(tmp_path))
    wandb_utils.init_wandb_primary(args)
    assert sdk.init.call_args.kwargs["id"] == "explicit"
    assert sdk.init.call_args.kwargs["settings"] == {"mode": "offline"}
    assert "resume" not in sdk.init.call_args.kwargs
    wandb_utils.init_wandb_secondary(args)
    assert sdk.init.call_args.kwargs["id"] == "explicit"
    assert sdk.init.call_args.kwargs["settings"] == {"mode": "offline"}
    sdk.login.assert_not_called()


def test_disabled_mode_never_logs_in_or_overwrites_identity(tmp_path, sdk):
    path = tmp_path / "wandb_run.json"
    path.write_text("existing")
    args = _args(wandb_mode="disabled", wandb_key="test-secret", save=str(tmp_path))
    wandb_utils.init_wandb_primary(args)
    wandb_utils.init_wandb_secondary(args)
    assert sdk.init.call_args.kwargs["settings"] == {"mode": "disabled"}
    assert path.read_text() == "existing"
    sdk.login.assert_not_called()


def test_secrets_are_redacted_without_changing_training_args():
    args = _args(
        wandb_key="wandb-value",
        service={"api_key": "key-value", "headers": {"Authorization": "bearer-value"}},
        hf_token="hf-value",
        rollout_max_response_tokens=4096,
        tokenizer="qwen",
    )
    config = wandb_utils._compute_config_for_logging(args)
    assert config["wandb_key"] == "[REDACTED]"
    assert config["service"]["api_key"] == "[REDACTED]"
    assert config["service"]["headers"]["Authorization"] == "[REDACTED]"
    assert config["hf_token"] == "[REDACTED]"
    assert config["rollout_max_response_tokens"] == 4096
    assert config["tokenizer"] == "qwen"
    assert args.service["api_key"] == "key-value"
    critic = wandb_utils._compute_secondary_config_for_logging(args, role="critic")
    assert critic["critic/wandb_key"] == "[REDACTED]"


def test_opd_metrics_use_the_coordinator_step(sdk):
    wandb_utils._init_wandb_common()
    sdk.define_metric.assert_any_call("opd/step")
    sdk.define_metric.assert_any_call("opd/*", step_metric="opd/step")
    sdk.define_metric.assert_any_call("perf/opd_*", step_metric="opd/step")
    calls = [call.args[0] for call in sdk.define_metric.call_args_list]
    assert calls.index("perf/opd_*") < calls.index("perf/*")


def test_wandb_remains_opt_in_for_generic_entrypoints(sdk):
    args = _args(use_wandb=False, use_opd=False, wandb_run_id="unused")
    wandb_utils.init_wandb_primary(args)
    sdk.init.assert_not_called()
    assert args.wandb_run_id is None


@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_launcher_wandb_environment(mode, tmp_path):
    helper = Path(__file__).resolve().parents[1] / "examples/on_policy_distillation/wandb-env.sh"
    env = {key: value for key, value in os.environ.items() if not key.startswith("WANDB_")}
    env.update(WANDB_MODE=mode, WANDB_ENTITY="team", WANDB_RUN_ID="chosen", SAVE_DIR=str(tmp_path))
    command = 'source "$1" fixture-group\nprintf "%s\\0" "${WANDB_ARGS[@]}"'
    result = subprocess.run(["bash", "-c", command, "bash", str(helper)], env=env, capture_output=True, check=True)
    args = result.stdout.decode().rstrip("\0").split("\0")
    assert "--use-wandb" in args
    assert args[args.index("--wandb-project") + 1] == "slime-mopd"
    assert args[args.index("--wandb-group") + 1] == "fixture-group"
    assert args[args.index("--wandb-mode") + 1] == mode
    assert args[args.index("--wandb-run-id") + 1] == "chosen"
    assert args[args.index("--wandb-team") + 1] == "team"
    assert args[args.index("--wandb-dir") + 1] == str(tmp_path / "wandb")
