"""Small MOPD resume state, separate from model/optimizer tensors."""

import json
import os
from pathlib import Path


def load_state(directory, rollout_id, config):
    if directory is None or rollout_id is None or rollout_id < 0:
        return None
    path = Path(directory) / "rollout" / f"opd_state_{rollout_id}.json"
    if not path.exists():
        return None
    with path.open() as stream:
        state = json.load(stream)
    if state.get("schema_version") != 1 or state.get("sampling_fingerprint") != config["sampling_fingerprint"]:
        raise ValueError("MOPD checkpoint teacher/domain configuration changed")
    return state


def save_state(directory, rollout_id, config, optimizer_steps, version_steps, latest_version, teacher_heads):
    path = Path(directory) / "rollout" / f"opd_state_{rollout_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1,
        "sampling_fingerprint": config["sampling_fingerprint"],
        "resolved_config": config,
        "optimizer_steps": optimizer_steps,
        "version_steps": version_steps,
        "latest_version": latest_version,
        "teacher_heads": teacher_heads,
    }
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(state, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
