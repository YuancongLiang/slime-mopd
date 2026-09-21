from __future__ import annotations

import asyncio
from types import SimpleNamespace

import e2b
import pytest

from slime.agent.sandbox import E2BSandbox

NUM_GPUS = 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("template_mode", "expected_template"),
    [(None, None), ("true", "swebench-example")],
)
def test_image_can_select_local_e2b_template(monkeypatch, template_mode, expected_template):
    monkeypatch.setenv("SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY", "image")
    if template_mode is None:
        monkeypatch.delenv("SLIME_AGENT_E2B_TEMPLATE_FROM_IMAGE", raising=False)
    else:
        monkeypatch.setenv("SLIME_AGENT_E2B_TEMPLATE_FROM_IMAGE", template_mode)

    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(sandbox_id="sandbox-id")

    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)

    async def enter():
        return await E2BSandbox("swebench-example", rpc_retries=1).__aenter__()

    asyncio.run(enter())

    assert captured["template"] == expected_template
    assert captured["metadata"] == {"image": "swebench-example", "size": "md"}
