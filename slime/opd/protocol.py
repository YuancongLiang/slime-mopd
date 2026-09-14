"""Versioned, pickle-free hidden transport and model identity checks."""

import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def request_id(teacher_id, version, tokens, response_length):
    return digest([teacher_id, version, tokens, response_length])


def model_identity(checkpoint):
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    config = getattr(config, "text_config", config)
    text = config.to_dict()
    for key in ("_name_or_path", "transformers_version", "torch_dtype", "dtype", "architectures"):
        text.pop(key, None)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenization = {"vocab": tokenizer.get_vocab(), "special_tokens": tokenizer.special_tokens_map}
    if getattr(tokenizer, "is_fast", False):
        backend = json.loads(tokenizer.backend_tokenizer.to_str())
        backend.pop("padding", None)
        backend.pop("truncation", None)
        tokenization["backend"] = backend
    return {
        "model_hash": digest(text),
        "tokenizer_hash": digest(tokenization),
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
    }


def verify_info(info, teacher_id, teacher, identity):
    if info.get("protocol") != 1 or info.get("teacher_id") != teacher_id or info.get("version") != teacher["version"]:
        raise ValueError("Teacher protocol/id/version mismatch")
    for key, value in identity.items():
        if info.get(key) != value:
            raise ValueError(f"Teacher {teacher_id} {key} mismatch: {info.get(key)!r} != {value!r}")
    if not isinstance(info.get("head_hash"), str) or len(info["head_hash"]) != 64:
        raise ValueError("Teacher response has no valid head hash")


def read_tensor_response(response, name, shape, max_bytes):
    import torch
    from safetensors.torch import load

    response.raise_for_status()
    data = bytearray()
    for chunk in response.iter_bytes():
        data.extend(chunk)
        if len(data) > max_bytes:
            raise ValueError("Teacher tensor exceeds its declared byte budget")
    tensors = load(bytes(data))
    if set(tensors) != {name} or tensors[name].shape != tuple(shape) or tensors[name].dtype != torch.bfloat16:
        raise ValueError(f"Invalid teacher {name} tensor shape/dtype")
    return tensors[name]
