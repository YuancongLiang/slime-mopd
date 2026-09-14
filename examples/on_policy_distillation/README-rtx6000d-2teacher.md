# Single-node RTX 6000D: Qwen3.5-2B with two teachers

This preset assumes seven 84GB GPUs, indexed 0 through 6. Run each model size separately. RTX 6000D kernel compatibility and end-to-end throughput still require validation on the target machine.

| Physical GPUs | Role | Parallelism |
| --- | --- | --- |
| 0-3 | Student learner | TP=1, DP=4 |
| 4 | Student rollout | TP=1 |
| 5 | Math teacher | TP=1 |
| 6 | Code teacher | TP=1 |

The YAML uses two example domains, `math` and `code`, sampled equally. Change `sampling_weight` to change their relative frequency. Both positive-weight domains must exist in `metadata.domain` in the prompt dataset. Teacher and student must use compatible Qwen3.5-2B configs/tokenizers.

Defaults: BF16, GBS=64, one response per prompt, 8192 total tokens, 4096 response tokens, 4096 packing budget, 128-token KL chunks, 3GiB teacher-head cache per learner, no rollout prefetch. A long trajectory may exceed the packing budget; the context cap remains 8192.

## Environment

Activate the Slime environment in every terminal. From the repository root, set actual paths:

```bash
export MEGATRON_ROOT=/root/Megatron-LM
export PYTHONPATH="$MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_CHECKPOINT=/models/Qwen3.5-2B
```

TileLang, Transformer Engine, FlashAttention/FLA and SGLang must support the installed GPU/CUDA combination. The `torch` OPD backend can be selected in YAML to isolate KL kernel issues; it does not replace these other dependencies.

## Start the two teachers

In separate terminals, keep both commands running:

```bash
TEACHER_CHECKPOINT=/checkpoints/math-rl \
  bash examples/on_policy_distillation/serve-rtx6000d-2teacher.sh math
```

```bash
TEACHER_CHECKPOINT=/checkpoints/code-rl \
  bash examples/on_policy_distillation/serve-rtx6000d-2teacher.sh code
```

The script fixes the corresponding GPU, ID, version and port to match the YAML. Versions must identify immutable checkpoints: when replacing a teacher, update its version in both the YAML and teacher script. `TEACHER_GPU` overrides the physical GPU index when adapting the layout; keep teacher GPUs outside the Ray mask.

Verify that both services are ready:

```bash
curl --fail http://127.0.0.1:7999/info
curl --fail http://127.0.0.1:8000/info
```

## Start a dedicated Ray head

Use an idle machine or reserve these GPUs first. Do not connect this preset to an existing Ray cluster exposing the teacher GPUs. The driver environment alone cannot change an already running Ray head's GPU visibility.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 ray start --head \
  --node-ip-address=127.0.0.1 --port=16379 --num-gpus=5 \
  --dashboard-host=127.0.0.1 --dashboard-port=18265 \
  --temp-dir=/tmp/slime-ray-2teacher
export RAY_ADDRESS=127.0.0.1:16379
```

The loopback endpoints deliberately restrict this preset to one node. Do not add other Ray nodes. No script stops existing Ray services automatically.

## Start the learner and rollout

Example dataset row:

```json
{"prompt":[{"role":"user","content":"Calculate 12 * 13."}],"metadata":{"domain":"math"}}
```

```bash
export STUDENT_CHECKPOINT=/checkpoints/student-init
export PROMPT_DATA=/data/prompts.jsonl
export SAVE_DIR=/checkpoints/mopd-2b-2teacher
bash examples/on_policy_distillation/run-rtx6000d-7gpu-2teacher.sh --finetune
```

For a smoke run, prefix the command with `NUM_ROLLOUT=2 GLOBAL_BATCH=8`. For resume, point `STUDENT_CHECKPOINT` to the saved MOPD checkpoint root and omit `--finetune`. GBS must be divisible by DP=4. `PACKED_TOKENS`, `GLOBAL_BATCH`, `NUM_ROLLOUT`, `SAVE_INTERVAL`, `LEARNING_RATE` and `OPD_CONFIG` can be overridden through environment variables. Additional CLI arguments are forwarded last.

When changing the context limit, update the YAML, teacher limits and learner/rollout arguments together. If the card fails TileLang compilation, run the existing OPD tests with `backend: torch` first and validate the installed toolchain before switching back.
