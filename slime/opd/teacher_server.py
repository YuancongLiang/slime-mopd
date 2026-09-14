"""Frozen, optimizer-free Megatron hidden server, one torchrun TP group per replica.

Launch with the model's normal Slime/Megatron arguments plus --debug-train-only.
The HTTP thread owns metadata/CPU tensors; only the main thread enters CUDA/NCCL.
"""

import hashlib
import queue
import threading
import time

from .protocol import request_id


class CapacityError(Exception):
    pass


class TargetStore:
    def __init__(self, info, max_bytes, max_requests, max_context, ttl):
        self.info = info
        self.max_bytes, self.max_requests, self.max_context, self.ttl = max_bytes, max_requests, max_context, ttl
        self.entries = {}
        self.reserved = 0
        self.lock = threading.Lock()
        self.pending = queue.Queue(maxsize=max_requests)

    def _prune(self):
        now = time.monotonic()
        for key, value in list(self.entries.items()):
            if value["state"] != "running" and now - value["created"] > self.ttl:
                self._remove(key)

    def _remove(self, key):
        entry = self.entries.pop(key, None)
        if entry is not None:
            self.reserved -= entry["bytes"]

    def submit(self, payload):
        info = self.info
        if payload.get("teacher_id") != info["teacher_id"] or payload.get("version") != info["version"]:
            raise ValueError("Teacher id/version mismatch")
        tokens, length = payload.get("tokens"), payload.get("response_length")
        if not isinstance(tokens, list) or not 2 <= len(tokens) <= self.max_context:
            raise ValueError("Invalid trajectory length")
        if any(type(t) is not int or not 0 <= t < info["vocab_size"] for t in tokens):
            raise ValueError("Invalid token id")
        if type(length) is not int or not 0 < length < len(tokens):
            raise ValueError("Response must have a nonempty prefix and at least one token")
        key = request_id(info["teacher_id"], info["version"], tokens, length)
        if payload.get("request_id") != key:
            raise ValueError("Request digest mismatch")
        required = length * info["hidden_size"] * 2 + len(tokens) * 40 + 65536
        if required > self.max_bytes:
            raise ValueError("A single target exceeds the server byte budget")
        with self.lock:
            self._prune()
            if key in self.entries:
                return key
            if (
                len(self.entries) >= self.max_requests
                or self.reserved + required > self.max_bytes
                or self.pending.full()
            ):
                raise CapacityError("Teacher queue/cache is full; defer target until learner consumption")
            canonical = {"tokens": tokens, "response_length": length}
            self.entries[key] = {
                "payload": canonical,
                "state": "queued",
                "bytes": required,
                "created": time.monotonic(),
                "result": None,
            }
            self.reserved += required
            self.pending.put_nowait(key)
        return key

    def next(self):
        while True:
            key = self.pending.get()
            with self.lock:
                if key in self.entries:
                    self.entries[key]["state"] = "running"
                    return key, self.entries[key]["payload"]

    def finish(self, key, result=None, error=None):
        with self.lock:
            entry = self.entries[key]
            entry.update(state="error" if error else "ready", result=result, error=error, created=time.monotonic())
            # The request token list is no longer needed once its digest identifies the result.
            entry["payload"] = None

    def get(self, key):
        with self.lock:
            self._prune()
            entry = self.entries.get(key)
            return dict(entry) if entry is not None else None

    def release(self, key):
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None and entry["state"] == "running":
                raise ValueError("Cannot release a target while its CUDA computation is running")
            self._remove(key)

    def stats(self):
        with self.lock:
            self._prune()
            states = {state: 0 for state in ("queued", "running", "ready", "error")}
            for entry in self.entries.values():
                states[entry["state"]] += 1
            return states | {"reserved_bytes": self.reserved, "max_bytes": self.max_bytes}


def build_app(store, head):
    import torch
    from aiohttp import web
    from safetensors.torch import save

    app = web.Application(client_max_size=16 * 1024**2)

    async def info(request):
        return web.json_response(store.info)

    async def loads(request):
        return web.json_response(store.stats())

    async def submit(request):
        try:
            key = store.submit(await request.json())
            return web.json_response({"request_id": key}, status=202)
        except CapacityError as exc:
            return web.json_response({"error": str(exc)}, status=429)
        except (ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def result(request):
        entry = store.get(request.match_info["key"])
        if entry is None:
            return web.Response(status=404)
        if entry["state"] == "error":
            return web.json_response({"error": entry["error"]}, status=500)
        if entry["state"] != "ready":
            return web.Response(status=202)
        return web.Response(
            body=entry["result"],
            content_type="application/octet-stream",
            headers={"X-OPD-Head": store.info["head_hash"]},
        )

    async def release(request):
        try:
            store.release(request.match_info["key"])
            return web.Response(status=204)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=409)

    async def get_head(request):
        try:
            start, count = int(request.query["start"]), int(request.query["count"])
            if start < 0 or count <= 0 or count > head.shape[0] or start > head.shape[0] + count:
                raise ValueError("Invalid head shard")
            shard = torch.zeros(count, head.shape[1], dtype=torch.bfloat16)
            valid = max(0, min(count, head.shape[0] - start))
            shard[:valid].copy_(head[start : start + valid])
            return web.Response(
                body=save({"head": shard}),
                content_type="application/octet-stream",
                headers={"X-OPD-Head": store.info["head_hash"]},
            )
        except (ValueError, KeyError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    app.router.add_get("/info", info)
    app.router.add_get("/loads", loads)
    app.router.add_post("/submit", submit)
    app.router.add_get("/result/{key}", result)
    app.router.add_delete("/result/{key}", release)
    app.router.add_get("/head", get_head)
    return app


def add_arguments(parser):
    parser.add_argument("--opd-server-id", required=True)
    parser.add_argument("--opd-server-version", required=True)
    parser.add_argument("--opd-server-host", default="0.0.0.0")
    parser.add_argument("--opd-server-port", type=int, default=7999)
    parser.add_argument("--opd-server-max-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--opd-server-max-requests", type=int, default=1024)
    parser.add_argument("--opd-server-max-context", type=int, default=32768)
    parser.add_argument("--opd-server-ttl", type=float, default=3600)
    return parser


def main():
    import os
    import torch
    import torch.distributed as dist
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model
    from safetensors.torch import save

    from slime.agent.aiohttp_threaded import run_app_in_thread
    from slime.backends.megatron_utils.checkpoint import load_checkpoint
    from slime.backends.megatron_utils.data import DataIterator, get_batch
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model_provider import get_model_provider_func
    from slime.opd.megatron import gather_hidden, head_forward, head_weight, unwrap
    from slime.opd.protocol import model_identity
    from slime.utils.arguments import parse_args
    from slime.utils.distributed_utils import init_gloo_group

    args = parse_args(add_custom_arguments=add_arguments)
    if (
        not args.debug_train_only
        or args.use_opd
        or args.pipeline_model_parallel_size != 1
        or args.context_parallel_size != 1
    ):
        raise ValueError("Hidden server requires --debug-train-only, OPD disabled, PP=CP=1")
    if args.expert_model_parallel_size != 1 or int(os.environ["WORLD_SIZE"]) != args.tensor_model_parallel_size:
        raise ValueError("Launch one TP group per server replica: WORLD_SIZE=TP, EP=1")
    if not args.bf16 or args.enable_mtp_training:
        raise ValueError("Hidden server requires BF16 and MTP disabled")
    if (
        min(args.opd_server_max_bytes, args.opd_server_max_requests, args.opd_server_max_context, args.opd_server_ttl)
        <= 0
    ):
        raise ValueError("Teacher server limits must be positive")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    init_gloo_group()
    args.rank, args.world_size = dist.get_rank(), dist.get_world_size()
    args.no_load_optim = args.no_load_rng = True
    args.finetune = True
    init(args)
    models = get_model(get_model_provider_func(args, "actor"), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    for model in models:
        model.requires_grad_(False)
        model.eval()
    load_checkpoint(models, None, None, {})
    model = models[0]
    module = unwrap(model)
    if module.output_layer.bias is not None:
        raise ValueError("Hidden server requires a bias-free head")
    local_head = head_weight(module).detach().contiguous()
    identity = model_identity(args.hf_checkpoint)
    head_parts = [torch.empty_like(local_head) for _ in range(args.world_size)] if args.rank == 0 else None
    dist.gather(local_head, head_parts, dst=0)
    store, handle = None, None
    if args.rank == 0:
        head = torch.cat(head_parts)[: identity["vocab_size"]].cpu().to(torch.bfloat16).contiguous()
        del head_parts
        head_hash = hashlib.sha256(memoryview(head.view(torch.uint8).numpy())).hexdigest()
        info = identity | {
            "protocol": 1,
            "teacher_id": args.opd_server_id,
            "version": args.opd_server_version,
            "head_hash": head_hash,
        }
        store = TargetStore(
            info,
            args.opd_server_max_bytes,
            args.opd_server_max_requests,
            args.opd_server_max_context,
            args.opd_server_ttl,
        )
        handle = run_app_in_thread(build_app(store, head), host=args.opd_server_host, port=args.opd_server_port)

    def hidden_forward(layer, input_, **kwargs):
        return gather_hidden(layer, input_), None

    try:
        while True:
            request = [store.next() if args.rank == 0 else None]
            dist.broadcast_object_list(request, src=0)
            key, payload = request[0]
            tokens = payload["tokens"]
            length = payload["response_length"]
            data = {
                "tokens": [torch.tensor(tokens, device="cuda", dtype=torch.long)],
                "total_lengths": [len(tokens)],
                "response_lengths": [length],
                "loss_masks": [torch.ones(length, device="cuda", dtype=torch.int)],
            }
            batch = get_batch(DataIterator(data, [[0]]), list(data), args.data_pad_size_multiplier)
            # Captures the final normalized prediction-head input, without executing the head GEMM.
            with torch.inference_mode(), head_forward(model, hidden_forward):
                hidden = model(
                    input_ids=batch["tokens"],
                    position_ids=None,
                    attention_mask=None,
                    labels=None,
                    packed_seq_params=batch["packed_seq_params"],
                    loss_mask=batch["full_loss_masks"],
                )
            if args.rank == 0:
                response_hidden = (
                    hidden[0, len(tokens) - length - 1 : len(tokens) - 1]
                    .to(device="cpu", dtype=torch.bfloat16)
                    .contiguous()
                )
                store.finish(key, save({"hidden": response_hidden}))
            del hidden
    finally:
        if handle is not None:
            handle.stop()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
