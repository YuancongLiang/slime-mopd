"""Launch the actual optimizer-free torchrun server with a tiny local HF checkpoint."""

import os
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest
import torch

from slime.opd.protocol import read_tensor_response, request_id

NUM_GPUS = 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tp", [1, 2])
def test_megatron_teacher_server_from_hf_checkpoint(tmp_path, tp):
    if torch.cuda.device_count() < tp:
        pytest.skip("Not enough CUDA GPUs")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    torch.manual_seed(15)
    reference = LlamaForCausalLM(config).to(dtype=torch.bfloat16)
    reference.save_pretrained(tmp_path)
    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="t0", bos_token="t1", eos_token="t2"
    ).save_pretrained(tmp_path)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={tp}",
        "-m",
        "slime.opd.teacher_server",
        "--debug-train-only",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(tp),
        "--tensor-model-parallel-size",
        str(tp),
        "--num-layers",
        "1",
        "--hidden-size",
        "64",
        "--ffn-hidden-size",
        "128",
        "--num-attention-heads",
        "4",
        "--group-query-attention",
        "--num-query-groups",
        "4",
        "--kv-channels",
        "16",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        "1e-5",
        "--swiglu",
        "--disable-bias-linear",
        "--position-embedding-type",
        "rope",
        "--rotary-base",
        "10000",
        "--vocab-size",
        "128",
        "--untie-embeddings-and-output-weights",
        "--hf-checkpoint",
        str(tmp_path),
        "--load",
        str(tmp_path),
        "--global-batch-size",
        "1",
        "--rollout-batch-size",
        "1",
        "--micro-batch-size",
        "1",
        "--num-rollout",
        "0",
        "--seq-length",
        "16",
        "--max-position-embeddings",
        "64",
        "--bf16",
        "--opd-server-id",
        "teacher",
        "--opd-server-version",
        "v1",
        "--opd-server-port",
        str(port),
    ]
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ
            | {
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "PYTHONPATH": os.environ.get("MEGATRON_ROOT", "/root/Megatron-LM")
                + ":"
                + os.environ.get("PYTHONPATH", ""),
            },
        )
    try:
        with httpx.Client(timeout=2, trust_env=False) as client:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("Teacher server failed to start:\n" + log_path.read_text()[-16000:])
                try:
                    info = client.get(url + "/info")
                    if info.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.5)
            assert info.json()["hidden_size"] == 64
            tokens, response_length = [4, 5, 6, 7, 8], 3
            key = request_id("teacher", "v1", tokens, response_length)
            payload = {
                "teacher_id": "teacher",
                "version": "v1",
                "request_id": key,
                "tokens": tokens,
                "response_length": response_length,
            }
            assert client.post(url + "/submit", json=payload).status_code == 202
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("Teacher prefill failed:\n" + log_path.read_text()[-16000:])
                with client.stream("GET", url + "/result/" + key) as result:
                    if result.status_code == 200:
                        hidden = read_tensor_response(result, "hidden", (3, 64), 65536)
                        break
                    assert result.status_code == 202
                time.sleep(0.1)
            with client.stream("GET", url + "/head", params={"start": 0, "count": 128}) as result:
                head = read_tensor_response(result, "head", (128, 64), 65536)
            reference.cuda().eval()
            with torch.inference_mode():
                expected = reference(torch.tensor([tokens], device="cuda")).logits[0, 1:4].float()
                actual = (hidden.cuda() @ head.cuda().t()).float()
            torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.03)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
