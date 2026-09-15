# OPD 性能与质量对照

目标是回答两个不同问题：全词表 OPD 的 head/loss 多花多少计算，以及相同资源预算下整个训练任务慢多少。指标同时用于 sampled 和 full-vocabulary 路径，原有训练、rollout 日志保持可用。

## W&B 与记录

`examples/on_policy_distillation/` 的 OPD 训练脚本默认启用 W&B，项目默认为 `slime-mopd`。通过 `WANDB_PROJECT`、`WANDB_ENTITY`、`WANDB_GROUP`、`WANDB_MODE=online|offline|disabled` 设置归属和联网方式。直接运行 Python 入口仍需 `--use-wandb`。续训使用 checkpoint 中记录的 run ID；新实验使用不同输出目录。

Online 模式需要各 Ray 节点上的训练用户完成 `wandb login`，或在启动各节点的 Ray 进程之前设置 `WANDB_API_KEY`。已经运行的远端 Ray worker 不会自动继承提交端后来新增的环境变量。

性能记录保存在 `${SAVE_DIR}/opd_metrics.jsonl`，可用 `--opd-metrics-dir` 独立指定目录。`--opd-metrics-warmup 2` 排除最初两批的稳态比较，`--opd-metrics-interval 20` 控制细粒度 GPU 计时的采样间隔，设为 0 只保留基础指标。基础计数与耗时每批记录，比较工具只读取 `opd/phase=steady`。恢复记录带有 checkpoint 边界标记，比较工具忽略崩溃前尚未保存的旧分支，原始日志仍保留。Offline 模式会生成可后续 sync 的日志片段，不等同于 online 原位续写。

主要口径：

| 数据 | 含义 |
| --- | --- |
| `opd/effective_tokens` | 本批用于成功更新的有效 response token，排除 prompt、padding 和 mask=0 |
| `opd/computed_tokens` | 本批 response 行数，包含 mask=0 的行；不重复计算 TP 副本或 backward 重算 |
| `perf/opd_learner_s` | 本批 learner 路径墙钟时间；包括实际暴露在该路径中的 teacher 等待 |
| `perf/opd_e2e_s` | 主控流水线的本批实际墙钟窗口；重叠阶段不相加 |
| `perf/opd_learner_compute_s` | 采样批次中参数更新的 GPU span，含 forward、loss、backward、梯度归并与 optimizer；不含更新前单独计算 log-prob/advantage 的阶段；各 rank 先合计再取最大值 |
| `perf/opd_head_loss_s` | 两种模式的 head/loss forward+backward GPU span，含全词表重算 |
| `perf/opd_peak_allocated_gib` | learner 最重 rank 的 PyTorch allocated 峰值，不等于整机显存 |

原有 `perf/actor_train_tok_per_s` 使用 prompt+response 口径，适合沿用历史曲线，但不替代有效蒸馏 token/s。GPU head/loss 细分、teacher 等待和缓存指标解释额外开销；异步计时重叠时不能简单用总时长减去各分项求“纯计算”。

定位 MOPD 额外成本时可添加以下 W&B 曲线：

| 指标 | 用途 |
| --- | --- |
| `perf/opd_head_loss_forward_s`、`perf/opd_head_loss_backward_s` | head/KL 的前向、反向耗时，反向包含必要的 logits 重算 |
| `perf/opd_teacher_prefill_s_mean`、`perf/opd_teacher_queue_s_p95` | teacher 主干平均耗时和排队长尾；各 teacher 另有 `perf/opd_teacher/<id>/...` 指标 |
| `perf/opd_target_wait_latency_s_mean`、`perf/opd_hidden_download_s` | learner 等待目标与下载 hidden 的时间 |
| `perf/opd_head_cache_hit_ratio`、`perf/opd_head_download_latency_s_mean` | 冻结 head 的缓存效果与 miss 下载延时 |
| `perf/opd_hidden_download_bytes`、`perf/opd_head_download_bytes` | hidden 和各 TP head 分片的实际传输量 |
| `perf/opd_kl_tp_reduce_s`、`perf/opd_hidden_tp_broadcast_s` | learner TP 内 KL 归约与 hidden 广播耗时 |
| `perf/opd_workspace_allocations`、`perf/opd_workspace_bytes_max` | Linear/KL 主工作区的复用情况及常驻容量，不含 TileLang 内部 scratch 缓存 |
| `perf/opd_policy_lag_steps_max`、`perf/opd_skipped_updates` | rollout 策略滞后和未成功更新的步数 |
| `perf/opd_domain/<domain>/kl_token_mean` | full-vocabulary 分领域有效 token 加权 KL |
| `perf/opd_domain/<domain>/sampled_kl_token_mean` | sampled 分领域 KL 估计 |

Teacher 服务的 `/loads` 还提供累计请求、token、传输字节、队列和最近 256 次耗时。Teacher 显存指标只报告该服务 TP rank 0 的 PyTorch 峰值；`d2h_wait` 包含等待此前 GPU 工作完成的时间，不能与 GPU prefill 耗时相加。

## 固定轨迹回放

固定初始 student/teacher、精度、GBS、TP/DP/EP、packing 和重计算设置，保存实际生成的轨迹。下面两个模式都回放同一批 tokens；单 teacher 可用原版 Megatron sampled 路径，双 teacher 使用新增的 SGLang 领域路由。

```bash
NUM_ROLLOUT=12 bash examples/on_policy_distillation/run-rtx6000d-7gpu-2teacher.sh \
  --save-debug-rollout-data '/data/opd-replay/{rollout_id}.pt' --finetune
```

上述命令沿用现有模型/数据/输出环境变量。严格单 teacher 比较时，采集数据仅含一个领域，MOPD YAML 对应领域路由至待比较的 teacher。文件保留 tokens、response 长度、loss mask 和 rollout log-prob；只加载可信的本地 PyTorch 文件。

回放通过 `--load-debug-rollout-data` 自动进入 `debug_train_only`，不启动 student 生成引擎；full-vocabulary hidden 服务仍需运行。脚本默认 4 卡 learner，两次都使用 `--finetune` 从同一初始权重开始。单 teacher 比较示例：

新脚本启用 `--opd-refresh-replay-targets`，使 sampled 回放重新取得当前冻结 teacher 的目标；原有 sampled debug 回放默认沿用已保存的数据。Full-vocabulary 回放始终刷新 hidden 目标。

固定回放的 full YAML 使用 `prefetch_rollouts: false`，与 sampled 主控顺序一致；仓库示例已采用此值。工具也会校验这项条件，防止把下一批预取的收益算成 head/KL 算法差异。

```bash
export HF_CHECKPOINT=/models/qwen3.5-2b-hf
export STUDENT_CHECKPOINT=/models/qwen3.5-2b-initial-mcore
export PROMPT_DATA=/data/prompts.jsonl
export TEACHER_CHECKPOINT=/models/math-rl-mcore
export OPD_CONFIG=/data/opd-single-math.yaml
export REPLAY_DATA='/data/opd-replay/{rollout_id}.pt'
export OPD_BENCHMARK_TEACHER_ID=math-rl-checkpoint-v1
export BENCHMARK_DIR=/data/opd-compare
SAVE_DIR="$BENCHMARK_DIR/sampled" bash examples/on_policy_distillation/run-opd-replay.sh sampled
SAVE_DIR="$BENCHMARK_DIR/full" bash examples/on_policy_distillation/run-opd-replay.sh full
python -m slime.opd.compare --sampled /data/opd-compare/sampled \
  --full /data/opd-compare/full --output /data/opd-compare/comparison.json
```

`--opd-benchmark-teacher-id` 是两次运行共享的不可变 checkpoint 标识，必须对应同一个 teacher 的实际权重；工具不读取整个 checkpoint 来验证用户填写的标识。对照工具检查每批 token/mask/domain 的 workload hash、训练条件、batch ID 和有效 token 数；条件不一致时拒绝输出受控对照。

双 teacher 时，使用 `examples/on_policy_distillation/sampled-2teacher.yaml` 配置 sampled 领域权重与 SGLang endpoint，权重和 teacher 版本必须与 full YAML 相同。SGLang server 返回 sampled log-prob，不能把 hidden server 的 7999/8000 端口直接填进去。两个 2B SGLang teacher 的启动示例（分别占 GPU 5、6，与 hidden teacher 分两轮运行）：

```bash
CUDA_VISIBLE_DEVICES=5 python -m sglang.launch_server --model-path /models/math-rl-hf \
  --host 127.0.0.1 --port 8999 --tp 1 --dtype bfloat16
CUDA_VISIBLE_DEVICES=6 python -m sglang.launch_server --model-path /models/code-rl-hf \
  --host 127.0.0.1 --port 9000 --tp 1 --dtype bfloat16
```

两个服务需要分别在独立终端启动，model-path 必须是对应冻结 teacher 的 HF 权重。设置 `SAMPLED_DOMAIN_CONFIG=examples/on_policy_distillation/sampled-2teacher.yaml` 后，`run-opd-replay.sh sampled` 自动使用领域路由；此时不需要 `TEACHER_CHECKPOINT`。两轮均设置 `OPD_BENCHMARK_TEACHER_ID=math-v1+code-v1`。在线 sampled 也可通过 `--opd-domain-config` 使用同一加权 sampler，保持原 sampled policy loss。

回放墙钟不包含 student rollout 生成，不能称为在线全流程吞吐。Teacher 的实际执行方式仍会影响结果：Megatron 基线在 learner 本地计算目标，full-vocabulary 通过 hidden 服务获取目标。请同时看 learner/GPU 细分、teacher 服务和传输指标，分别解释算法计算量与部署成本。冷启动、首次 head 下载及编译单独看 warmup 数据；稳态中真实发生的缓存 miss、重试仍属于成本。

`perf/opd_e2e_s` 排除显式 checkpoint 等待，评估在该窗口外执行；`perf/opd_job_effective_tok_per_s` 从任务开始计时，包含初始化、保存与已完成的评估。比较 `learner` 墙钟时仍应注意不同 teacher 路径的等待位置。`learner_compute` 用于定位参数更新开销，不包含 sampled 路径单独计算旧 log-prob 的成本；回答整轮慢多少应以 `e2e` 为准，并用 learner 分项和 head 微基准解释差异。

## 在线对比与报告

在线运行使用相同 GPU 预算、领域权重和长度上限，通常生成不同轨迹。此时显式允许不同 workload：

```bash
python -m slime.opd.compare --sampled /data/online-sampled --full /data/online-full \
  --allow-different-workloads --output /data/online-comparison.json
```

报告中 `matched_workload=false` 表示结果混合了生成长度、数据与实现差异。吞吐按 `sum(tokens) / sum(seconds)` 计算，不平均每批 token/s；显存使用观察窗口内最大值。`time_multiplier=1.25` 表示同量 token 耗时增加 25%，对应 `throughput_drop_percent=20`，这两个百分比不同。

## 独立 Head/Loss 微基准

```bash
python -m slime.opd.benchmark --objective both --backend tilelang \
  --rows 512 --hidden-size 2048 --vocab-size 248320 --chunk-tokens 256 --iterations 20
```

两种模式使用相同 BF16 hidden/head 和 FP32 head 梯度缓冲。sampled 基线使用 Slime 的 log-prob/entropy 与 clipped policy loss，以 `-(old_logp-teacher_logp)` 作为 OPD advantage，task reward 和 entropy coefficient 均为零；不是 hard-label CE。标签从固定 student 分布抽样，teacher sampled log-prob 提前计算，其 head/log-prob 准备耗时另报。full-vocabulary 每次执行 student/teacher head、KL 和 backward 重算。所有计时均在预热之后。

此工具只量化 response 行上的 head/loss 成本，不包含主干、prompt head、optimizer、rollout、网络或多 teacher 调度，不能外推为整轮训练慢多少。`--objective both|sampled` 目前要求 TP=1；保留默认 full-vocabulary 的 `torchrun` TP 微基准。随机 hidden 下的 KL 只用于数值诊断，不是模型质量评估。

## 蒸馏质量

保留现有 loss、entropy、梯度、策略滞后和 skipped update；新增分领域有效 token 份额与 KL，检查长回答领域是否占据过多训练量。sampled 的 `logp_student-logp_teacher` 是采样估计，允许负值，不能与训练中的精确全词表 KL 当作同口径曲线。

能力合并使用现有分领域评测入口和相同生成设置验证。训练日志中的在线 KL 下降不等价于固定评测集上的能力提升。本次未新增 held-out 全词表 KL 评估器；后续比较分布距离时，应让两种训练产物在同一组固定前缀、同一个 teacher 和相同温度下计算精确全词表 KL。
