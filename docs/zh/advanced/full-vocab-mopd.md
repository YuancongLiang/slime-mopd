# 全词表 MOPD：实现与运行

本页描述已经接入的接口。早期设计文档中的嵌套 YAML 仍是方案示意；实际配置参见 [full-vocab.yaml](../../../examples/on_policy_distillation/full-vocab.yaml)。

## 实现范围

- `train_mopd.py`：独立 rollout、learner、外部 teacher 三池，可选单批 rollout 预取。checkpoint 批次不预取下一批，避免保存超前的采样 cursor。
- `slime.opd.teacher_server`：每个冻结 teacher 副本一个独立 `torchrun` TP group。使用 Megatron 主干，不创建 optimizer/DDP、不执行 LM head，返回最终 norm 后的 response 预测位置 hidden。
- `slime.opd.linear_kl`：自定义 autograd 覆盖两端 head 和 `KL(student || teacher)`，按 token chunk 重算，teacher 不求梯度。
- `slime.opd.tilelang_kl`：SM80+ FP32 行统计与 dZ kernel，显式输出 buffer。动态行数避免为每种尾块长度重新编译；GEMM 使用 PyTorch/cuBLAS，TP 使用 NCCL。
- 领域加权抽样、teacher 多副本、head 缓存、二进制 hidden、版本/shape/tokenizer 校验、策略滞后与恢复状态。
- 旧 RL/SFT/sampled OPD 默认路径保留；未启用时不加载新 kernel 或创建 teacher 服务。

## 环境与部署

使用 Slime 环境，并让完整 Megatron 源码可导入，而不只有 `megatron.core`。例如在启动 Ray 和 teacher 之前设置：

```bash
export MEGATRON_ROOT=/root/Megatron-LM
export PYTHONPATH="$MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

本机验证环境为 PyTorch 2.11.0+cu129、TileLang 0.1.11、A100-SXM4-80GB。选择 `backend: tilelang` 时需要与 CUDA 环境匹配的 TileLang；选择 `backend: torch` 可使用分块参考实现，不导入 TileLang。普通 sampled OPD 不增加此依赖。

主布局：两个互通 IB 的节点分别放 8 卡 learner 和 8 卡 rollout；两个 TCP 节点运行 teacher 服务。teacher 不需要加入训练 Ray 集群，也不共享 learner WORLD。若 teacher 节点加入 Ray，不要重复分配已被独立 teacher 占用的 GPU。

YAML 可选 `placement.learner_nodes` / `rollout_nodes` 使用实际 Ray 节点 IP，限制两个资源池放置；每个选中节点分配相同 GPU 数。不配置时沿用 Slime 原资源分配方法。teacher 的节点、GPU mask 和 TP 由独立服务启动命令决定。

## 启动 Teacher

在 teacher 节点、仓库根目录运行一个 2B 副本，checkpoint 可为 HF 或 Megatron 权重：

```bash
export MODEL_PRESET=scripts/models/qwen3.5-2B.sh
export HF_CHECKPOINT=/models/qwen3.5-2b
export TEACHER_CHECKPOINT=/checkpoints/math-rl
export TEACHER_ID=math_rl
export TEACHER_VERSION=math-step-1000
export TEACHER_TP=1
export TEACHER_PORT=7999
CUDA_VISIBLE_DEVICES=0 bash examples/on_policy_distillation/serve-hidden.sh
```

增加 teacher 使用不同 ID/版本、checkpoint、端口与 GPU mask；增加同一 teacher 的副本保持 ID/版本/checkpoint 相同，将副本 URL 加入其 `endpoints`。版本必须不可变。TP=2 时使用例如 `CUDA_VISIBLE_DEVICES=0,1`，一个副本不能跨 TCP 节点。

接口为 `GET /info`、`GET /loads`、`POST /submit`、`GET /result/{id}`、`DELETE /result/{id}`、`GET /head?start=...&count=...`。hidden/head 使用 safetensors 二进制，不传 JSON 浮点数组或 pickle。

默认每副本最多 1024 个请求、约 8GiB 请求/目标预算、32K 上下文和 3600 秒 TTL；对应 `--opd-server-max-requests`、`--opd-server-max-bytes`、`--opd-server-max-context`、`--opd-server-ttl`。模型、CPU head、计算工作区和序列化临时内存另计。队列满返回 429，rollout 延迟获取目标，learner 消费时重新提交，避免整个 global batch 等待缓存腾空。单请求超预算直接拒绝，不截断序列。

## 配置与数据

`teachers` / `domains` 不限制为 7 个；每领域指向一个 teacher，可设非负有限权重。按权重比例随机抽样，不是每步严格等额配额，也不再次将采样权重乘到 loss 上。所有正权重领域必须有样本。

```json
{"prompt":[{"role":"user","content":"计算 12 × 13。"}],"metadata":{"domain":"math"}}
```

默认读取 `metadata.domain`，用 `opd.domain_key` 更换字段名。使用 Slime 默认开启的 global dataset，不传 `--disable-rollout-global-dataset`。

每轨迹只有一个 teacher。hidden 仅由负责样本的 learner TP rank 0 从网络获取，再在 TP group 内广播；teacher TP 可以不同于 learner TP。同一 microbatch 内按 teacher 聚合 response 行，再重建词表分片。有效词表从 HF text config 读取，仅排除 Megatron 对齐 padding。

loss 按完整 rollout 的有效 response token 数归一化，再沿用 Slime 的全局 batch/梯度累积缩放。跨片段的完整 rollout 保持 `rollout_mask_sums` 分母。

## 启动训练

Ray 集群和所有 teacher 服务就绪后，从仓库根目录运行：

```bash
export MODEL_PRESET=scripts/models/qwen3.5-2B.sh
export HF_CHECKPOINT=/models/qwen3.5-2b
export STUDENT_CHECKPOINT=/checkpoints/student-init
export OPD_CONFIG=examples/on_policy_distillation/full-vocab.yaml
export PROMPT_DATA=/data/prompts.jsonl
export SAVE_DIR=/checkpoints/mopd-2b
bash examples/on_policy_distillation/run-full-vocab.sh --finetune
```

示例 YAML 中域名和版本须替换为实际服务信息。首次从既有 RL checkpoint 开始新 MOPD 任务可加 `--finetune`；续训指向 MOPD checkpoint，并移除该参数。

| 模型 | Learner 起测设置 | Teacher 起测 |
| --- | --- | --- |
| 2B | `LEARNER_TP=1 PACKED_TOKENS=8192` | TP=1 |
| 9B | 使用 9B preset，`LEARNER_TP=2 PACKED_TOKENS=4096`，脚本末尾加 `--sequence-parallel` | TP=1 |
| 剪枝 19B-A3B | 真实剪枝 preset/config；候选 `LEARNER_TP=1 LEARNER_EP=8 PACKED_TOKENS=4096` | 节点内 TP=2，需实测 |

新增 2B preset 对应[官方 config](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json)，保留共享 embedding。19B 不用原 35B 的层数/专家数冒充剪枝结构；实际 config 尚需提供，不能声称已验证真实 19B 训练。

默认 GBS=128、每 prompt 一条 rollout、总长 32K、response 上限 16K。`PACKED_TOKENS` 是 packing budget，不保证单条超长轨迹显存可行。`head_cache_bytes` 必须容纳一个 microbatch 涉及的冻结 head 分片，否则报错。

`prefetch_rollouts: true` 开启单批预取。`max_policy_lag_optimizer_steps` 默认 1，检查实际 optimizer step 滞后；发布版本另有映射。轨迹不能混用 student 版本。一个 rollout batch 执行多个 optimizer step 时需重新设置上限。

模型/optimizer 保持原 checkpoint 格式，额外状态写入 `rollout/opd_state_<rollout_id>.json`，记录更新计数、发布版本映射、teacher head hash 和配置。领域 RNG/cursor/buffer 随 global dataset 状态保存。修改 endpoint/性能参数可迁移；更改 teacher 版本或领域采样语义拒绝精确恢复。评估继续使用评估数据集自己的 reward 配置或 `--rm-type`，不调用 hidden reward hook。

## 验证与测量

```bash
python -m pytest tests/test_full_vocab_opd.py tests/test_opd_runtime.py tests/test_opd_tp.py \
  tests/test_opd_megatron_integration.py tests/test_opd_training_step.py tests/test_opd_teacher_server.py
python -m slime.opd.benchmark --rows 512 --hidden-size 2048 --vocab-size 248320 --chunk-tokens 256
torchrun --standalone --nproc-per-node=2 -m slime.opd.benchmark --hidden-size 4096 --chunk-tokens 512
```

基准包含两个 head、KL、重算、student dH/dW，预热后计时并报告显存。它不包含主干、rollout、teacher prefill、网络与权重发布，不能当作完整训练 tokens/s。

本机 A100 已验证 dense 对照、TileLang、TP/SP、Megatron `main_grad`、共享/非共享 head、真实独立 teacher 服务。尚无四节点网络、真实三个模型训练或 H100 实测。H100 按真实 CUDA target 编译同一 kernel，不自动开启 FP8。

本次验证共 92 项通过：34 项新增 MOPD 测试（含 TP=1/2 独立服务）及 58 项既有回归测试；另通过 Ruff、启动脚本语法和 `git diff --check`。独立服务测试使用临时生成的小型 HF 模型，梯度集成测试使用小型 Megatron GPT，不能替代真实 Qwen 混合注意力/MoE 模型的端到端验收。

首版要求纯文本、BF16、无 bias 的 head、PP=CP=1；拒绝 VPP/MTP/routing replay/FP8/partial rollout 等未适配组合。teacher 每副本常驻一个主干，数量受 GPU 容量限制，没有 CPU/NVMe 主干换入换出。权重发布复用原实现，尚非双 snapshot 完全异步 publisher。固定 workspace 覆盖大型 OPD vocabulary 临时张量，不代表整个框架零动态分配。
