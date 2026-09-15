"""Start frozen-teacher prefill while subsequent student rollouts are generated."""

import asyncio
import time

import aiohttp

from .protocol import request_id


async def reward_func(args, sample, **kwargs):
    if isinstance(sample, list):
        return await asyncio.gather(*(reward_func(args, item, **kwargs) for item in sample))
    # Evaluation retains its dataset-specific reward hooks; this hook has no task reward.
    cfg = args.opd_resolved
    domain = (sample.metadata or {}).get(cfg["domain_key"])
    if domain not in cfg["domains"]:
        raise ValueError(f"No MOPD teacher configured for domain {domain!r}")
    teacher_id = cfg["domains"][domain]["teacher"]
    teacher = cfg["teachers"][teacher_id]
    has_media = any(value is not None for value in (sample.multimodal_inputs or {}).values())
    if has_media or sample.multimodal_train_inputs:
        raise ValueError("Full-vocabulary MOPD currently supports text only")
    if len(sample.tokens) > cfg["max_context_tokens"]:
        raise ValueError("MOPD trajectory exceeds max_context_tokens; refusing silent truncation")
    endpoint = teacher["endpoints"][(sample.index or 0) % len(teacher["endpoints"])].rstrip("/")
    target = {
        "teacher_id": teacher_id,
        "domain": domain,
        "version": teacher["version"],
        "endpoint": endpoint,
        "request_id": request_id(teacher_id, teacher["version"], sample.tokens, sample.response_length),
    }
    payload = target | {"tokens": sample.tokens, "response_length": sample.response_length}
    started = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=cfg["timeout_seconds"])) as session:
        async with session.post(endpoint + "/submit", json=payload) as response:
            # A full teacher cache must not stall completion of the global rollout batch.
            # The learner resubmits deferred requests as it consumes and releases earlier targets.
            if response.status != 429:
                response.raise_for_status()
                result = await response.json()
                if result.get("request_id") != target["request_id"]:
                    raise ValueError("Teacher returned a mismatched request id")
            deferred = int(response.status == 429)
    sample.metadata = sample.metadata or {}
    sample.metadata.setdefault("opd_metrics", {}).update(
        teacher_submit_s=time.perf_counter() - started, teacher_deferred=deferred
    )
    sample.opd_target = target
    return 0.0
