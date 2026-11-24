"""Pinned official mini-SWE-agent shared by distillation, RL and evaluation.

Only the model transport and remote environment are adapters. The agent loop,
action parser, observation renderer and submission detector are upstream code.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import yaml

from .common import BaseHarness

UPSTREAM_TAG = "v2.0.0"
UPSTREAM_COMMIT = "ce91a8c6449bc5a686df8f6b7ee8ca9433d893bc"


def load_official():
    root = Path(os.environ["SLIME_AGENT_MINI_ROOT"]).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"mini-SWE-agent must be pinned at {UPSTREAM_COMMIT}; found {commit}")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src/minisweagent"], cwd=root).returncode:
        raise RuntimeError("Pinned mini-SWE-agent source has local changes")
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    import minisweagent

    if not Path(minisweagent.__file__).resolve().is_relative_to(root):
        raise RuntimeError("A different mini-SWE-agent package is already imported")
    config_path = root / "src/minisweagent/config/benchmarks/swebench.yaml"
    config = yaml.safe_load(config_path.read_text())
    return config, {
        "upstream_tag": UPSTREAM_TAG,
        "upstream_commit": commit,
        "package_version": minisweagent.__version__,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "cost_limit": 0,
        "cost_limit_note": "Gateway/local inference prices are unspecified; USD cap disabled, time/steps/tokens enforced.",
        "temperature": "omitted for v2 per official leaderboard guidance",
    }


def public_messages(messages):
    return [{k: v for k, v in message.items() if k != "extra"} for message in messages if message.get("role") != "exit"]


class OfficialRun:
    """Bridge official synchronous components to the caller's asyncio loop."""

    def __init__(self, sb, backend, *, workdir, time_budget, step_limit=250, folder=None):
        self.sb, self.backend, self.workdir = sb, backend, workdir
        self.deadline = time.monotonic() + time_budget
        self.step_limit, self.folder = step_limit, folder
        self.config, self.fingerprint = load_official()
        self.loop = asyncio.get_running_loop()
        self.cancelled = threading.Event()
        self.pending = set()
        self.counter = 0
        self.agent = None

    def call(self, factory):
        if self.cancelled.is_set():
            raise RuntimeError("mini-SWE-agent interrupted")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("mini-SWE-agent time budget exhausted")
        future = asyncio.run_coroutine_threadsafe(factory(), self.loop)
        self.pending.add(future)
        try:
            return future.result(timeout=remaining + 1)
        except FutureTimeout:
            future.cancel()
            raise TimeoutError("mini-SWE-agent time budget exhausted") from None
        finally:
            self.pending.discard(future)

    async def execute(self, action):
        from slime.agent.sandbox import exec_and_wait

        number = self.counter
        self.counter += 1
        timeout = min(self.config["environment"]["timeout"], max(1, int(self.deadline - time.monotonic())))
        command = action.get("command", "")
        if not isinstance(command, str):
            return {"output": "", "returncode": -1, "exception_info": "bash command must be a string"}
        env = self.config["environment"].get("env", {})
        cmd = f"cd {shlex.quote(self.workdir)} && source .harness/env.sh && timeout --kill-after=5s {timeout}s bash -c {shlex.quote(command)} 2>&1"
        rc, output = await exec_and_wait(self.sb, cmd=cmd, user="agent", env=env, time_budget_sec=timeout + 15, tag=f"mini_{number}", want_output=True)
        if self.folder:
            (self.folder / f"tool_{number:03d}.log").write_text(output)
        return {"output": output, "returncode": rc, "exception_info": "Command timed out" if rc in {124, 137} or rc < 0 else ""}

    def build(self):
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
        from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages, parse_toolcall_actions

        owner = self
        config = copy.deepcopy(self.config)

        class Environment(DockerEnvironment):
            def __init__(self):
                self.container_id = None
                env_config = {k: v for k, v in config["environment"].items() if k != "environment_class"}
                self.config = DockerEnvironmentConfig(image=owner.sb.image, **env_config)

            def execute(self, action, cwd="", *, timeout=None):
                output = owner.call(lambda: owner.execute(action))
                self._check_finished(output)  # official marker + successful exit required
                return output

        class Model:
            def query(self, messages, **kwargs):
                message = owner.call(lambda: owner.backend.query(public_messages(messages), owner.deadline))
                tool_calls = [SimpleNamespace(id=c["id"], function=SimpleNamespace(**c["function"])) for c in message.get("tool_calls", [])]
                actions = parse_toolcall_actions(tool_calls, format_error_template=config["model"]["format_error_template"])
                return {**message, "extra": {"actions": actions, "cost": 0.0, "timestamp": time.time()}}

            def format_message(self, **kwargs):
                return kwargs

            def format_observation_messages(self, message, outputs, template_vars=None):
                return format_toolcall_observation_messages(actions=message.get("extra", {}).get("actions", []), outputs=outputs, observation_template=config["model"]["observation_template"], template_vars=template_vars)

            def get_template_vars(self, **kwargs):
                return config["model"]

            def serialize(self):
                return {"info": {"pinned_harness": owner.fingerprint, "model_transport": type(owner.backend).__name__}}

        agent_config = config["agent"] | {"cost_limit": 0, "step_limit": self.step_limit}
        if self.folder:
            agent_config["output_path"] = self.folder / "mini_trajectory.json"
        self.agent = DefaultAgent(Model(), Environment(), **agent_config)
        return self.agent

    async def run(self, task):
        agent = self.build()
        worker = asyncio.create_task(asyncio.to_thread(agent.run, task))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            self.cancelled.set()
            for future in list(self.pending):
                future.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(worker), 10)
            except (Exception, asyncio.CancelledError):  # noqa: BLE001 - drain the sync worker on cancellation
                pass
            raise


class SlimeChatBackend:
    def __init__(self, url, sid):
        self.url, self.sid = url, sid

    async def query(self, messages, deadline):
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL

        payload = {"model": "slime-actor", "messages": messages, "tools": [BASH_TOOL], "parallel_tool_calls": True, "stream": False}
        timeout = aiohttp.ClientTimeout(total=max(1, deadline - time.monotonic()))
        async with aiohttp.ClientSession(timeout=timeout) as client:
            async with client.post(self.url + "/v1/chat/completions", json=payload, headers={"Authorization": f"Bearer {self.sid}"}) as response:
                if response.status != 200:
                    raise RuntimeError(f"Slime adapter returned {response.status}: {(await response.text())[:500]}")
                body = await response.json()
        return body["choices"][0]["message"]


class MiniSweHarness(BaseHarness):
    name = "mini_swe"

    async def install_cli(self, sb):
        # Runs on the host; the sandbox needs only its prepared Bash/Python env.
        load_official()

    async def write_config(self, sb, ctx):
        pass

    async def launch_and_wait(self, sb, ctx, prompt, time_budget_sec):
        task = await sb.read_file(f"{ctx.workdir}/PROBLEM_STATEMENT.md", user="agent")
        run = OfficialRun(sb, SlimeChatBackend(ctx.adapter_url, ctx.session_id), workdir=ctx.workdir, time_budget=time_budget_sec, step_limit=int(os.environ.get("SLIME_AGENT_MINI_STEP_LIMIT", "250")))
        result = await run.run(task)
        data = run.agent.serialize()
        data["info"]["pinned_harness"] = run.fingerprint
        await sb.write_file(f"{ctx.workdir}/.harness/mini_trajectory.json", json.dumps(data), user="agent")
        await sb.write_file(f"{ctx.workdir}/.harness/mini_submission.patch", result.get("submission", ""), user="agent")
        return 0 if result.get("exit_status") == "Submitted" else -1
