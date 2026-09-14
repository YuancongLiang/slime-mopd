# 全词表多教师 OPD 合版设计分析

状态：原始设计分析。首版代码、实际配置和本机 A100 验证见 [实现与运行](full-vocab-mopd.md)；本文保留数学推导与扩展设计。

部署方案更新：用户进一步确定使用 4 台 8 卡 A100、独立 rollout/learner/teacher 三池，其中两个节点仅 TCP。最新拓扑、初始训练参数和 TileLang 实施细节见 [32 卡三资源池规划](full-vocab-mopd-32gpu-plan.md)。本文保留为数学和同构模型接入分析；下文 learner 内轮换 teacher 权重是共置后端，不代表最新部署主方案。

代码基线：`3778dbf6d1a533ab478ecf5ddaa11449a47752b2`，分析日期：2026-09-11。

## 1. 已确认的目标与建议

本次需求是：student 与多个 teacher 使用同一型号；teacher 是从 student 经不同领域 RL 得到的专家；每条样本按领域选择一个 teacher，通过领域采样权重控制合版比例。需要三个独立规模的纯文本任务：Qwen3.5-2B、Qwen3.5-9B、从 Qwen3.6-35B-A3B 剪枝得到的 19B-A3B，每个任务可能有 7 个或更多 teacher。不是把三个不同规模的 checkpoint 加载进同一个模型实例。

建议采用以下执行链路：

```text
领域加权抽取 prompt，记录 domain / teacher_id
    -> 当前 student 生成轨迹
    -> 同构 Megatron 模型按 teacher 分组切权重、执行无梯度前向
    -> 缓存监督位置对应的 teacher 最终 hidden states
    -> 恢复 student 权重
    -> 读取 teacher hidden 和冻结的 teacher head 词表分片
    -> 在 student TP 组内计算分块全词表 reverse KL 并反向
    -> 各领域共同完成一个 optimizer step
```

核心选择：保留全词表目标，缓存 hidden 而不是整个 rollout 的 logits；词表维保持 TP 分片；纯合版使用独立可微 KL loss；按实际显存预算决定多个 teacher head 常驻还是按阶段换入。

当前机器可见 8 张 A100-SXM4-80GB，主机总内存约 1.0 TiB。模型规模与至少 7 个 teacher 已确认，19B 剪枝配置暂不可得，按用户指定使用原 Qwen3.6-35B-A3B 的 LM head 尺寸。上下文与 response 长度、实际启动脚本及是否使用其他节点尚未确定；本文容量数字是静态计算，不是训练实测结果。具体模型预算见第 10.4 节。

## 2. 论文依据与工程推演的边界

提供的 Hugging Face PDF 链接在本次访问中返回 404。已核对可访问的 [DeepSeek-V4 技术报告](https://arxiv.org/html/2606.19348v1#S5.SS1.SSS2) 第 5.1.2 节和 [第 5.2.2 节](https://arxiv.org/html/2606.19348v1#S5.SS2.SSS2)。

报告采用学生轨迹上的全词表 reverse KL，按任务选择领域专家。工程上缓存教师末层 hidden、训练时重建 logits，结合教师分组、异步加载和专用 KL kernel。

下面的 TP 梯度推导、Slime 接口、缓存布局、head 驻留策略和分阶段实现都是针对本仓库的设计建议。报告没有提供可直接搬入 Slime 的完整实现，也没有给出本机上的加速比。

## 3. 现有代码的实际路径

| 位置 | 当前行为 | 对改造的影响 |
| --- | --- | --- |
| `slime/rollout/on_policy_distillation.py:8` | SGLang 请求同一条学生 token 序列的 input token logprobs | 当前 HTTP 返回字段只有 sampled-token 监督，不能恢复全分布 |
| `slime/rollout/on_policy_distillation.py:34` | 裁剪 response 的 teacher logprobs，纯蒸馏返回零任务 reward | 全词表路径不应继续把重型张量作为 reward JSON 搬运 |
| `slime/backends/megatron_utils/actor.py:119` | 一个 `self.model`，不同 tag 保存 CPU 权重并切换 | 同构 RL teacher 可以复用实例和已有模型并行布局 |
| `slime/backends/megatron_utils/actor.py:414` | teacher 前向得到 `teacher_log_probs` | 需要独立的 hidden-only 前向和多 teacher 调度 |
| `slime/backends/megatron_utils/loss.py:663` | sampled `student_logp - teacher_logp` 减到 advantage | full-vocab 必须绕过这条注入路径，避免重复蒸馏 |
| `slime/backends/megatron_utils/loss.py:808` | OPD 注入后还可能执行 advantage whitening | 新 KL 不应被 advantage whitening 或 PPO clip 改写 |
| `slime/backends/megatron_utils/loss.py:933` | 最终通过 policy loss 更新 | 纯 MOPD 应增加独立 loss 分支 |
| `slime/backends/megatron_utils/model_provider.py:208` | `parallel_output=True` | student 原有 logits 已沿词表分片，不必 gather 全词表 |
| `slime/utils/ppo_utils.py:187` | TP softmax 使用 custom autograd | 可复用通信和手写 backward 模式，不能直接复用 sampled 目标 |
| `slime/utils/tensor_backper.py:21` | 每个 tag、每个 rank 分配 pinned CPU 权重，末尾同步 | 多教师扩展前必须估算 CPU 重复副本和切换带宽 |
| `slime/utils/dp_schedule.py:127` | 先确定 optimizer step，再 pack、分配 DP | teacher 分组应发生在既定 step 或目标准备窗口内 |

另一个隐含成本：`actor.py:401` 把 teacher 准备放在 `compute_advantages_and_returns` 条件内部。直接关闭 advantage 计算会同时跳过 teacher。需要先把目标准备抽出，再为纯 MOPD 跳过不必要的 reference、old-policy logprob 和 advantage 前向。

## 4. 训练目标与领域权重

设领域为 d，目标采样概率为 w_d，领域对应冻结教师为 q_d。对学生生成的轨迹，在监督位置 t 定义：

```text
p_t = softmax(z_student,t / T_student)
q_t = softmax(z_teacher(d),t / T_teacher)
KL_t = sum_{v in valid_vocab} p_t(v) * [log p_t(v) - log q_t(v)]

L = lambda * E_{d ~ w, x ~ D_d, y ~ rollout_student}
                  [sum_t mask_t * KL_t / sum_t mask_t]
```

这里先采用与仓库 per-rollout reduction 一致的语义：每条 rollout 的有效 assistant tokens 求均值，再跨 rollout 求均值。若一条 rollout 被切成多个训练 sample，需要沿用 `rollout_mask_sums`，不能分别按各片段长度归一化后重复计权。

- 领域按 w_d 采样后，不再给 loss 乘一次 w_d，否则实际权重变为近似 w_d 的平方。
- 若为调度采用不同的实际领域概率 q_d，才考虑以 w_d/q_d 校正，并记录实际抽样概率。这里的 q_d 是采样概率，与上面的教师分布含义需在代码中使用不同名称。
- 域权重定义在 prompt/rollout 层。若使用全局 token mean，长 response 领域将获得更大权重；这必须作为显式选项，而不是无意改变。
- 初版建议蒸馏温度均为 1，单独配置，不能自动复用 `rollout_temperature`。不隐式乘温度平方；若以后需要，明确作为目标定义的一部分。
- rollout 可以配置采样截断，但 KL 使用完整有效词表，不能复用当前 top-p replay mask。基线使用 temperature=1、top-p=1、top-k=-1，便于解释 on-policy 分布。
- 轨迹视为已采样的常量，直接对当前位置的完整条件分布反向。它不是对离散轨迹采样过程求导，也不应宣称等价于完整序列 KL 的所有状态分布梯度项。

纯 MOPD 不需要 GRPO advantage 或 PPO ratio。若未来要混合任务 RL，可显式定义 `L_total = L_RL + lambda * L_full_vocab_KL`，同时关闭 sampled OPD advantage 注入；RL 的 clipping/TIS 等策略如何作用于附加 KL 应另行定义。

## 5. Teacher 目标准备与缓存协议

### 5.1 同构模型复用

teacher 与 student 共用结构和运行时 TP/PP/CP/EP 配置，按 teacher id 在同一个 `self.model` 中切换参数。teacher 只前向，不建立梯度图，不加载 teacher optimizer 状态。

checkpoint 写入时的切分可以与运行时不同，但需先验证 Megatron distributed checkpoint 的可重分片性。建议把各 teacher 预处理为固定部署布局，避免每轮从网络 checkpoint 重新转换和重分片。

每次 teacher 前向完，必须完成当前 PP/VPP schedule、等待使用参数的 CUDA stream 结束，才能覆盖参数。所有 teacher 完成后恢复 student 参数及被切换的运行时 buffer；optimizer 的 FP32 master weights、动量和 step 仍属于 student。

teacher 的 MoE 路由应由各自权重计算，不能误用 student rollout 的 routed-expert replay 数据。纯 MOPD 还需处理当前依赖 old-policy record forward 的 routing replay 配置。

### 5.2 缓存什么

缓存的是生成目标 token 所需的预测头输入：完成最终 norm、必要的结构变换之后，进入 teacher LM head 之前的 hidden。不是任意 decoder 中间层，不是 KV cache，也不假设 hidden 比对本身构成蒸馏目标。

必须复现整个预测头，包括真实输出权重、可选 bias、logit scale/softcap、词表 padding 和量化解释。对于标准线性 head，就是 `h @ W_teacher.T + bias`；若模型有额外变换，统一封装在 head adapter 内。

序列 `prompt + response` 的第一个 response token 由最后一个 prompt 位置的 hidden 预测。因此监督位置需左移一位，复用现有 `full_loss_masks`/CP offset 定义。多轮工具输出和用户消息保留在前向上下文中，但通常不作为 KL 监督位置。

建议缓存协议至少包含：

```text
teacher_id, immutable_teacher_version
student_rollout_version
sample_id / rollout_id / segment_id
tokens_hash, prediction_positions
hidden_dtype, hidden_width, head_adapter_version
layout: TP/SP/CP ownership + packing description
buffer_handle, offset, length
```

token hash/position 信息用于防止重新 packing、截断和重试后取到错位目标。teacher version 必须与 head version 一致。采样后的完整 token ids 直接传 teacher，不进行 decode 再 tokenize。

### 5.3 存储与生命周期

- 初版先在既定 optimizer step 内构建单 teacher 的 packed microbatch；同一步包含多个 teacher 的 microbatch。目标准备和 student 使用这些相同的最终 packing 布局，按明确的 sample/position 回填，降低 CP 重排复杂度。一个 packed microbatch 内混合 teacher 的优化放在后续，届时需要显式位置重分布。
- cache store 存 BF16 hidden 和轻量索引；Ray 只传元数据/句柄，不广播 `[R,H]` Python 列表或 `[R,V]` JSON。
- 只存有效监督位置可节省容量；teacher 主干仍需计算完整 prompt 和 response 上下文。
- 采用有容量上限的窗口，最后 PP stage 按需预取到 GPU。不能把整个数据集的 hidden 永久缓存，也不能对所有 rank 无差别复制。
- CPU 主存可用 pageable/shared memory 存储，有限 pinned staging 用于双缓冲；大量长期 pinned allocation 会挤压系统内存。
- 同一组完整轨迹用于少量多个 optimizer step 时，可重用冻结 teacher 的 hidden；跨 rollout 不可只按 prompt 命中，因为 student 生成的前缀已经变化。
- cache 在最后一次使用及 backward/recompute 完成后释放，重启时校验版本与布局，过期项可重算。

当前 `forward_only` 是先执行模型输出再运行 callback。因此仅替换 callback 为“取 hidden”仍可能先计算完整 logits。必须在模型最后输出路径加 hidden-only adapter；同时不能简单把整个模型 `post_process=False`，以免跳过必要的最终 norm 或破坏 head 参数注册。

## 6. TP 分片的精确 KL 与 backward

设 TP 大小为 P，rank r 持有有效词表子集 V_r，对同一批 C 个 token：

```text
student_logits_r: [C, V_local]
teacher_logits_r: [C, V_local]
teacher_head_r:   [V_local, H_teacher]
```

同构模型无需 hidden 对齐投影，但仍必须核验 tokenizer、token id、有效词表和 padding 规则一致。teacher head 以 student 的 vocab shard 范围读取；PP 和 EP 不自动再分割这些 dense head 参数。

### 6.1 稳定的两次行统计归约

下面先以温度为 1 表示，两个模型的行 max 都是跨 TP 的全局 max：

```text
m_s = max_r max_v z_s,r(v)
m_t = max_r max_v z_t,r(v)             # 两组 [C] 可打包为一次 MAX all-reduce
a_r = z_s,r - m_s
b_r = z_t,r - m_t

S = sum_r sum_v exp(a_r(v))
Q = sum_r sum_v exp(b_r(v))
A = sum_r sum_v exp(a_r(v)) * (a_r(v) - b_r(v))
                                            # S/Q/A 打包为一次 SUM all-reduce
mu = A / S
KL = mu - log(S) + log(Q)
p_r = exp(a_r) / S
dL/dz_s,r(v) = upstream_weight * p_r(v) * [(a_r(v)-b_r(v)) - mu]
```

这是沿全有效词表求和的精确条件 reverse KL，在所选浮点精度内计算。TP 通信仅为每块的 O(C) 行统计，不需要 all-gather `[C,V]` logits。相同 KL 在各 TP rank 上用于对应词表分片的 backward，不额外除以 TP。

forward 保存行 max、S 和 mu 等 O(R) 统计后，backward 重算 logits 可直接使用这些统计构造局部梯度，无需再次执行上述 KL 行统计的 TP 通信。此处不包含 head 对 hidden 的必要梯度通信。

若 student 温度为 T_s，对缩放后的 logits 执行以上计算，最终对原 logits 的梯度再乘 `1/T_s`。teacher 固定，teacher logits/head/hidden 均不返回梯度。

可选优化：teacher 的 log-normalizer 对 student 是常量，在 student 梯度中完全抵消。因此只求梯度时可跳过 teacher exp 和归一化；但输出真实 KL、告警或 KL 自适应系数仍需它。初版保留真实 KL，后续再实测这一优化。

### 6.2 必须处理的数值情况

- BF16 存 hidden/head，GEMM 按硬件支持选择精度；max、exp、累加、KL 和梯度统计使用 FP32。
- 仅为分布式对齐新增的 padded vocabulary 行从 softmax 和差值求和中显式剔除。Qwen 原 checkpoint 定义的输出词表为 248320，不能因为 `tokenizer.vocab_size` 更小就删去模型原有输出行。两边均为 `-inf` 时相减会得到 NaN，不能寄希望于概率为零后消失。
- 相同 teacher/student 应得到接近零的 KL 和梯度；微小负 KL 可由舍入产生，日志可监控容差，不能用粗暴 clamp 掩盖实现错误。
- 不对有效 teacher 概率默默加 epsilon 或做 top-k 截断后仍称 exact KL。有限 raw logits 用稳定 logsumexp，避免先求低精度概率再取 log。
- 不把 `F.kl_div` 参数顺序当作理所当然；需明确计算 `KL(student || teacher)`，不是普通 soft-label cross entropy 的 forward KL。

### 6.3 Head 梯度同步

对于线性 head，自定义 student head 还需要正确计算；如果 head adapter 带有 scale/softcap，须先计算它们对 logits 的链式导数：

```text
dW_s,r = dZ_s,r.T @ H_s
dH_s,r = dZ_s,r @ W_s,r
dH_s   = sum_r dH_s,r
```

SP 关闭时，最后一式一般对应 TP SUM all-reduce。SP 开启时，head 前按 token 维 all-gather，backward 对 hidden 梯度 reduce-scatter。尽量在 microbatch 边界执行一次 hidden 通信，token chunk 内只做小型行统计归约。

不能在保留 `ColumnParallelLinear` 自带同步的同时再手工同步，也不能替换成普通 `F.linear` 后漏掉同步。参数梯度需要兼容 Megatron 的 `main_grad`、梯度累加 fusion、distributed optimizer 和 tied embeddings 梯度合并。

## 7. 分块计算能节省多少显存

需要区分三个实现级别：

| 级别 | 方式 | 显存边界 |
| --- | --- | --- |
| 小规模正确性基线 | 对 teacher/student full logits 直接计算 KL | O(R*V/TP)，仅用于验证 |
| 只分块 teacher/loss | 保留 student 原 output layer，teacher 分块重建 | 仍持有完整 student logits，且普通 autograd 会保存各块中间值 |
| 生产目标 | 从 student hidden 开始融合/封装 linear + KL，backward 分块重算 | logits 临时工作区 O(C*V/TP)，另外仍有 hidden、参数、梯度和主干激活 |

当前 `--recompute-loss-function` 只 checkpoint loss，输入的 student 完整 logits 仍存在；`ppo_utils` 的 chunk 参数也不能单独保证整步 O(C*V/TP) 显存。

建议采用第三种结构：forward 保存 student/teacher hidden 和每行少量统计；backward 用固定版本的 head 分块重建 logits，再计算 `dZ`、`dH`、`dW`，避免保存所有 chunk 的 softmax。需要显式 adapter 接入本地 Megatron 的 `LinearCrossEntropyModule`，已有 fused CE 不等于 fused reverse KL。

这项优化有代价：与保留 logits 相比，backward 会多做 student/teacher head 的重算 GEMM；它换取更大的 microbatch 和更少的显存流量，最终吞吐收益必须实测。相对现有 sampled OPD，teacher 原本也要执行 head；hidden-only 阶段跳过它后，在 student 阶段首次重建并非额外再算一次 teacher 主干。

初始可扫 `C=256/512/1024/2048`，同时观察 GEMM 利用率和 TP collective 延迟。块太小会使 kernel 启动和通信延迟占主导。A100 上先用 BF16 GEMM + FP32 统计；不照搬针对更新硬件的 FP4/FP8 路径。

## 8. CP、SP、PP、EP 和 DP 的布局约束

### CP：必须按预测位置对齐

本仓库同时有 zigzag/ring CP 和 `--allgather-cp` 的连续切片布局。`loss.py:98` 处理 response 左移对齐；`loss.py:194` 把连续 CP 的一维结果转换回下游需要的 zigzag 顺序。

teacher hidden 不能用 `response_length / CP` 简单均分。应按全局 prediction position 找到归属 rank，并记录真实 packing。特别是 `allgather_cp` 的 rank 归属依赖整个 packed microbatch；teacher 重新按领域 packing 后不能直接复用原 student 的局部索引。

第一版固定 teacher 目标与 student 的 packing；若重排，则显式重分布 hidden 或建立全局位置索引。KL 在局部 hidden/logits 上算完后，只对 `[R]` 结果使用现有 CP 重排；不跨 CP 重建 `[R,V]`。

SP 可以让 teacher hidden 的 token 维自然分布在 TP ranks。保留这种布局能减少缓存副本，但 head 前仍需 gather 对应的 token 行。只缓存有效监督行时，各 SP rank 行数通常不同，不能直接执行原有等长 sequence all-gather。须保存各 rank 长度和 prediction positions，pad 或使用不等长通信后恢复一致行序；也可先缓存原始 padded SP 网格，以空间换实现简单。student 的监督位置梯度应先 scatter 回原 packed 网格，再 reduce-scatter。每 CP 组内单 owner 保存后分发也是选项，但要把副本数计入容量。

### PP/VPP：仅最终输出 stage 需要 teacher head

只在真正 `post_process` 的模型 chunk 上缓存最终 hidden、加载 head 和计算 KL。非最后 PP stage 不需要 teacher head；最后 stage 的 head 显存不能再除以 PP。

额外 teacher head GEMM 与 KL 会增加最后 stage 的计算负载。PP 较大时应实测 stage 时间，并考虑最后 stage 减少 transformer 层数，不能仅按层数平均切分后假设流水线仍平衡。

### EP：teacher 切换必须覆盖实际通信组

MoE 的 EP 组可能跨越通常的 DP 维度。如果组内不同 rank 使用不同 teacher，expert all-to-all 会调用混合 checkpoint 的专家。

初版采用全局确定的 teacher phase：相关 TP/PP/CP/EP ranks 一起进入同一 teacher，全部前向完成后切换。不能让每个 DP worker 任意独立切权重。EP 只分 expert 参数，dense attention、norm 和 head 的复制关系需单独计数。

### DP：领域分组不能破坏均衡和归一化

在已确定的 optimizer step 内按 teacher 分桶，再按长度/FLOPs packing、分给各 DP rank，保留 `rollout_id` 和原始 `partition`。

每个 phase 必须满足有关调度要求：相同 TP 组的 token chunk 数和顺序一致，PP/VPP microbatch 数符合原调度约束。稀有领域可能不足以填满所有副本，需要预先配额、较大的准备窗口或零 loss dummy batch；dummy 不计入样本数、领域权重或归一化，但仍需走必要的 backward collectives。

不能在 reward 归一化前随意重排原始 samples：当前 GRPO 处理和 pass@k 统计有按 prompt group 相邻排列的假设。即使初版纯 MOPD，也应保留这些身份信息，避免破坏通用 rollout 逻辑。

## 9. 多 head 常驻还是严格单 head

Teacher 主干和 teacher head 的生命周期要分开。主干按 teacher 批量前向后可以换出；head 在 student backward 重算时仍会使用。

同型号不代表各 RL teacher 的 head 权重相同。不能把 teacher head 保存为指向 `self.model` 的 view；恢复 student 时会覆盖它。tied embeddings 需通过模型的真实 shared-weight 接口取得后独立保存，保持教师版本不可变。

### 推荐的首个生产路径：有预算的多 head 缓存

在当前训练步所需 head 能放入预算时，保留这些冻结 head 的 TP 分片，直到相关 backward 全部完成。student 使用一次完整的混合领域训练调度，初版每个 packed microbatch 只有一个 teacher，同一步可以交错不同 teacher 的 microbatch，避免为换 head 频繁清空流水线。

缓存必须对 in-flight backward pin 住版本；达到预算时不能直接驱逐仍被 autograd 引用的 GPU tensor。预取和驱逐以明确事件及引用计数管理。该策略尤其适合 head 占用远小于模型主干的场景。

可以进一步检查各 teacher head 的内容是否完全相同；仅在权重、bias、dtype 和 logits 后处理配置一致时共享只读快照。某些 RL 配置冻结 head 时可能获益，但不能直接假定 7 个 teacher 的 head 相同，尤其 tied embedding 也会随输入 embedding 训练而变化。

### 显存受限的扩展路径：严格单 head 分阶段累积

```text
一次 zero_grad
    teacher A 数据：student F/B，流水线 drain
    换 teacher head
    teacher B 数据：student F/B，流水线 drain
    ...
一次梯度归约/finalize、clip、optimizer.step、scheduler.step
```

简单把 microbatch 按 teacher 排序并不足够：1F1B/VPP 在边界处可能同时保留旧 teacher 的 backward 和新 teacher 的 forward。严格单 head 需要排空边界，或者重新设计可重载的 backward；后者会增加加载次数。

不能对每个领域直接调用现有 `train_one_step`，因为它会独立 zero_grad 和 optimizer.step，将目标改成依次领域更新。需要拆开函数的累积与更新边界，并处理 Megatron schedule 自动触发的 `finalize_model_grads_func`、DDP bucket/no_sync 和统一的全局 loss 分母。

`loss.py` 对 Megatron microbatch 缩放有显式补偿。分块 schedule 如果传入 M_d 个 microbatch，closure 的补偿要与该次 schedule 一致，但样本或 token 归一化分母仍属于整个 optimizer step，不能每个领域各取一次均值后等权相加。

两种策略优化的是同一个领域混合目标。若多个 head 本来只占少量显存，强制单 head 可能因每个领域增加 PP bubble 而降低吞吐，应以端到端实测选择。

## 10. 容量与吞吐估算

令 R 为一个目标准备窗口中的有效监督 token 总数，V 为有效词表大小，H 为 teacher head 输入宽度，b 为缓存元素字节数。

```text
完整 teacher logits：R * V * b
teacher hidden：      R * H * b
压缩倍数：            V / H
单个 teacher head：   V * H * b
每个最后 PP stage 的 TP rank head：约 V_padded * H * b / TP
```

一条样本只选择一个 teacher 时，hidden 总量是 `sum_d R_d*H = R*H`，不乘 teacher 数 K。若采用每条样本多教师 ensemble 才会增加该项；这不属于当前需求。

示例假设 `V=131072, H=4096, R=32768, BF16=2 bytes`：

| 对象 | 容量 |
| --- | --- |
| 窗口全部 teacher logits | 8 GiB |
| 窗口全部 teacher hidden | 256 MiB |
| 一个完整 teacher head | 1 GiB |
| TP=4 时每个 head shard | 256 MiB |
| TP=4 时 8 个 head shard 常驻同一最终 stage rank | 2 GiB |
| TP=4、CP=2 且监督 token 均衡时局部完整 teacher logits | 1 GiB/rank |
| TP=4、chunk=512 时一个 BF16 logits 临时块 | 32 MiB/rank |

最后一项只是单个 buffer；student、teacher、FP32 统计临时区、hidden 和梯度都需额外计入。CP 按完整上下文划分，监督位置可能集中到一个 rank，上述局部 logits 示例最坏可达 2 GiB；峰值应使用 `max(R_cp)` 而不是固定除以 CP。缓存 1,048,576 个监督 token 时，hidden 已达 8 GiB，仍必须有容量上限和及时释放。

head 常驻开销按该 stage 的 TP 分片算；CP 和 DP 通常复制 head，不把二者作为分母。hidden 若 SP 分布存储可减小每 rank 容量，若 gather 后每 TP rank 都缓存则会重复 TP 份，必须明确实现。

### 10.1 权重备份成本

当前每 rank 的 pinned 权重备份会随 teacher 数线性增加。正确估算为 `K * sum_over_ranks(actual_local_parameter_bytes)`，而不是把总参数量依次除以 TP/PP/CP/EP/DP。

示例：8B 参数的 BF16 权重约 16 GB，8 个 teacher 单个逻辑副本就约 128 GB，尚未包含 student、ref、optimizer 和 hidden。dense 模型若有多个 DP/CP 副本，现有备份方式还会重复；MoE 则需按 dense/expert 各自复制组核算。

建议先实现有界 teacher weight LRU，后端为本地缓存或共享存储，仅少量 pinned staging。规模需要时再增加同节点 shard 去重、owner 分发或类似 ZeRO 的存储分片。所有异步操作都需要独立 staging buffer；不能在旧 teacher 尚在计算时向同一 GPU 参数地址预取新 teacher。

### 10.2 时间模型

共置且复用同一 GPU 参数实例时，阶段基本串行：

```text
t_window ≈ t_rollout
         + sum_d(t_load_teacher_d + t_teacher_backbone_on_R_d_contexts)
         + t_restore_student
         + t_student_forward_backward
         + t_teacher_head_and_KL
         + t_recompute_heads
         + t_unhidden_IO_and_pipeline_bubbles
```

teacher 主干需处理完整上下文长度，不能用 R 直接替代所有前向 token。单领域路由下，把所有领域的 teacher 工作加起来通常接近一次完整数据窗口的 teacher 前向量；教师数增加主要还会增加权重切换、尾部不均衡和流水线排空成本。

切换的下界由每 rank 权重字节数、实测 H2D 带宽及节点聚合带宽决定，不能套用单卡理论 PCIe 带宽乘 GPU 数。当前 `TensorBackuper` 的同步点也会暴露这部分延迟。

只有独立资源池且预取/队列足够时，稳定流水才可能接近 `max(rollout, teacher, learner)` 的瓶颈阶段时间。共用同一个模型实例的方案不能声称三个阶段自动完全重叠。

更大的目标准备窗口可摊薄 K 次权重切换，但增加 hidden 占用、等待时间和 rollout 策略滞后。初版同步 rollout，记录 student version，限制每批轨迹更新次数；异步扩展再设定最大 policy lag。

### 10.3 应测量的指标

核心指标是有效监督 tokens/s、端到端 rollout-step 时间和达到相同评测质量的 GPU-hours，不能只比较 KL kernel 时间。

同时记录 teacher load/backbone/head、student F/B、TP collectives、hidden H2D/D2H、PP bubble、head cache hit、CPU pinned 峰值、GPU 峰值、各领域有效样本比例及真实 KL。领域分桶会降低 packing 利用率，需在最终吞吐中计入。

### 10.4 三个目标模型、7 个以上 teacher 的实际预算

已核对官方 [Qwen3.5-2B config](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json)、[Qwen3.5-9B config](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json)、[Qwen3.6-35B-A3B config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json)。它们的文本输出词表均为 248320。19B 模型 head 按用户指定沿用原 35B-A3B；不推定剪枝后的层数、专家数或专家宽度。

| 模型 | head 输入 H | BF16 完整 head | 7 个 head，TP=1 | 7 个 head，TP=2 | 7 个 head，TP=4 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B | 2048 | 0.947 GiB | 6.631 GiB | 3.315 GiB | 1.658 GiB |
| Qwen3.5-9B | 4096 | 1.895 GiB | 13.262 GiB | 6.631 GiB | 3.315 GiB |
| 剪枝 19B-A3B | 2048 | 0.947 GiB | 6.631 GiB | 3.315 GiB | 1.658 GiB |

TP 表格是 head 容量算式，不代表所有列都是各模型可用或建议的全模型 TP 配置。7 以上按实际 K 线性放大；head 只需冻结权重，无需 teacher optimizer/梯度/master-weight。2B 官方配置 tied embeddings，teacher head 快照也必须与 student 可更新 embedding 分离。

对 32768 个有效监督 token，三个模型的完整 BF16 teacher logits 都是 15.156 GiB。2B/19B 的 hidden 仅 128 MiB，9B 为 256 MiB，分别缩小 121.25 倍和 60.625 倍。这个数量级支持以 hidden cache 为主线；不能沿用旧 Qwen 约 15 万词表的预算。

仅一个 teacher head forward 的运算量就约为每 token 1.017 GFLOPs（H=2048）或 2.034 GFLOPs（H=4096）。这还没有 student head、反向或重算；尤其 2B 和 A3B 激活规模模型上，head 并非可以忽略的成本。需要同时优化 GEMM 块大小和 KL 访存，不能只优化 softmax kernel。

按型号标称总参数做 BF16 粗估，7 个 teacher 单逻辑副本的主干及 head 权重约为：2B 任务 28 GB、9B 任务 126 GB、19B 任务 266 GB。实际应从纯文本 checkpoint 张量计数，包含/不包含 vision、共享参数去重及剪枝会使标称值与实际不同。A3B 是激活规模，不能据此把 19B 的单 teacher 存储算为 6 GB。

### 10.5 Qwen 当前实现限制与 8 卡候选配置

这几个模型使用混合 Gated DeltaNet 和 full attention。本仓库现有 `slime_plugins/models/qwen3_5.py` 的 GDN 投影是普通 `nn.Linear`；`hf_attention.py:107` 先 gather SP，`:117` 再 gather CP，在各相关 rank 上重复执行完整 GDN 后切回局部输出。它不是原生 TP/CP 分片的 GDN。因此：

- 加大 TP 会减少 head/MLP 等部分的分片大小，但不能按 TP 倍数减少 GDN 参数、计算和中间激活。
- 增加 CP 不能保证 GDN 的长序列显存随 CP 下降，反而增加完整 hidden 的 gather 和重复计算；长上下文需单独 profile GDN。
- 本地 Megatron 虽有原生 GDN，当前实现拒绝 `packed_seq_params`，不能直接替换现有 varlen plugin。实现原生 TP/CP GDN 是独立工程项，不应假装全词表 KL 改动本身会解决它。
- A100 使用 `--qwen-gdn-backend fla`；现有 `qwen_gdn_backend.py:12` 要求 FlashQLA 为 SM90 或更新硬件，本机 SM80 不适用。
- 当前 Qwen spec 不支持 `pipeline_model_parallel_layout` 自定义布局；`--allgather-cp` 在本仓库也限于指定 DSA 路径。通用 KL 可设计支持两类 CP，但这几个 Qwen 的首版不应启用不受支持的分支。

以下是 8 卡全部可用于该阶段、rollout 资源可按共置机制卸载时的基准候选，不是尚未实测的最佳配置：

| 任务 | 首选基准候选 | 比较候选与原因 |
| --- | --- | --- |
| 2B | TP=1、PP=1、CP=1、EP=1，普通 DP=8 | 对比 TP=2、DP=4；TP=1 避免 GDN 在 TP 上重复，7 个 head 约 6.6 GiB/rank，可先测全常驻 |
| 9B | TP=2、PP=1、CP=1、EP=1，普通 DP=4 | 对比 TP=4、DP=2；7 个 head 分别约 6.6/3.3 GiB/rank，但更大 TP 会减少 data replicas 并放大 GDN 重复成本 |
| 19B-A3B | TP=1、PP=1、CP=1、EP=8、expert-TP=1 | 对比 TP=2、EP=4、expert-TP=1；先用 EP 分散专家权重与计算，具体合法性取决于剪枝后专家数和 MCore 分组约束 |

EP 与普通 DP/TP 不是可以全部相乘的独立 GPU 轴，不能把 TP=1、DP=8、EP=8 解释为 64 张卡。实际使用 Megatron 构造出的 attention/expert 并行组检查权重所有权和通信。

优先从 PP=1 开始是因为 7 个以上 teacher 的分阶段前向容易增加 PP bubble，且这些 head 的大小在 A100 80GB 上有机会同时驻留。是否最终可放下，还取决于 student optimizer 分片、激活、sequence length 和缓存预算。长序列 OOM 后不能盲目加 CP，需要检查 GDN 的完整序列中间量；PP=2、checkpointing、减小 token budget 是需要分别实测的选项。

这些候选也暴露 CPU 备份的复制问题：按 nominal dense 权重估算，2B、DP=8 的 7 个 teacher 约 224 GB，9B、TP=2/DP=4 即便理想分片也约 504 GB；实际 GDN 在 TP 复制会进一步抬高 9B 的数值。这不是小量 pinned allocation，不能把本机 1 TiB 主存视为可以无限复制。19B 的 experts 经 EP 分片、GDN/dense 参数仍有复制，应逐张量统计，不能直接套 dense 的 DP 倍数。

剪枝 19B 的最终适配还需真实 `num_hidden_layers`、`layer_types`、`num_experts`、`num_experts_per_tok`、expert/shared-expert FFN 宽度及权重形状。LM head 可按现已确认的尺寸先实施。若剪枝改变 attention/GDN 层顺序，需核对 SGLang 与 Megatron 对 `layer_types` 的解释；若剪专家，需核对各层专家数、router 和专家切片。7 个 teacher 必须都继承同一剪枝结构，不能只凭总参数同为 19B 就混用。

本地有具体的构层风险：SGLang 的 `Qwen3NextConfig.layers_block_type` 按 `full_attention_interval` 重建周期，而 Slime 的 Qwen spec 可读取显式 `layer_types`。如果剪枝删除任意 decoder 层并打乱周期，应先修正两端解释或验证周期仍保持一致，否则即使 shape 能加载，rollout 和训练也可能不是同一模型。仓库已有 9B 和原 35B-A3B preset，2B 可依据官方文本 config 新增；19B 主干 preset 需在实际剪枝配置到位后生成。

纯文本训练使用 `text_config` 和文本模型路径，不需要构造或切换 vision tower；三个规模分别维护 teacher registry、采样状态和预算配置。MTP 辅助训练默认关闭，其监督与主 next-token OPD 不是同一个目标。

## 11. 建议的接口与实施顺序

配置以 manifest 描述领域与 teacher，具体参数名在实现时与现有 parser 对齐。以下仅是设计示意，当前代码不能直接运行这些字段：

```yaml
opd:
  objective: full_vocab_reverse_kl
  coefficient: 1.0
  student_temperature: 1.0
  teacher_temperature: 1.0
  chunk_tokens: 512
  target_cache_dtype: bf16
  head_residency: budgeted
  teachers:
    math:
      checkpoint: /checkpoints/math_rl
      version: math-rl-step-1000
    coding:
      checkpoint: /checkpoints/coding_rl
      version: coding-rl-step-1500
  domains:
    math:
      teacher: math
      sampling_weight: 0.4
    coding:
      teacher: coding
      sampling_weight: 0.6
```

加载时验证所有 teacher 的结构、tokenizer、有效 vocab、特殊 token、head shape 和版本；缺失 domain/teacher 映射立即报错。采样器 checkpoint 同时保存 RNG、各域 cursor、抽样计数和配置版本。

| 模块 | 计划改动 |
| --- | --- |
| `slime/utils/arguments.py` | 区分 sampled/full-vocab objective，解析 teacher manifest、cache/head预算和温度，校验组合 |
| `slime/rollout/data_source.py` | 按领域权重抽 prompt，保存可恢复的采样状态，整个 prompt group 继承 teacher |
| `slime/utils/types.py`、`slime/ray/rollout.py` | 传递 teacher/version/position 元数据；只分发 cache handle |
| `slime/utils/dp_schedule.py` | 既定 step 内 teacher-aware packing、并行对齐、稀有领域处理 |
| `slime/backends/megatron_utils/actor.py` | teacher registry、teacher phase、隐藏目标准备、纯 OPD 跳过旧 RL 前向 |
| `slime/backends/megatron_utils/model_provider.py` 及新 head adapter | hidden-only 输出、冻结 teacher head 快照、fused student head 接入 |
| `slime/backends/megatron_utils/model.py` | forward-only cache 收集、训练输出和 loss 接口；单 head 模式再拆累积边界 |
| `slime/backends/megatron_utils/loss.py` 及新 KL kernel 模块 | 独立 full-vocab KL、TP backward、CP reduction、掩码和全局归一化 |
| 新 target cache / teacher weight store 模块 | 有界存储、异步预取、版本及引用生命周期 |
| `slime/observability` | 各领域 KL、阶段时间、cache/带宽/显存指标 |

建议按以下里程碑推进，每阶段都保证蒸馏目标不变：

1. 正确性基线：两个同构 teacher，领域路由与权重，TP=1 的 dense reverse KL，单个混合领域 optimizer step。仅用于确认目标和元数据，不作为长序列交付方案。
2. 生产核心：TP 分片 KL、hidden-only teacher、版本化 hidden cache、linear+KL 分块重算、预算内多个 frozen head，先覆盖 PP=1/CP=1，再验证 SP/CP/PP。
3. 多教师扩展：teacher weight 有界缓存、按领域目标准备、MoE EP 同步、异步 staging。按观测到的瓶颈优化存储分片或 head 驻留。
4. 显存需要时增加严格单 head 的梯度累积调度；之后再拓展 VPP、MTP、复杂 routing replay 和异步 policy lag。未经验证的组合在配置校验时报错，不宣称默认支持。

“同型号”省去了异构 teacher 构造和不同 vocab TP 重排，但不省去上述 token 对齐、SP/CP 和 optimizer 梯度同步问题。

## 12. 验收标准

- 数学：dense FP64 reference 与分片/分块实现的 KL、student logits/hidden/head 梯度对齐；teacher 无梯度；共同平移不变性；相同模型近零。
- 数值：不同 chunk、padding vocab、极端有限 logits、BF16 vs FP32、空监督位置、全 mask 样本；真实 vocab 和 padding 分开验证。
- 对齐：首个 response token、EOS、截断、多轮工具 mask、packing 后样本重排、CP 边界、allgather CP 空 rank。
- 分布式：TP=1/2/4、SP 开关、CP=1/2、PP=1/2 的一轮梯度及参数更新一致性；MoE EP 使用正确 teacher；空领域、dummy 和 VPP 对齐不死锁。
- MOPD：领域权重不重复乘；teacher version/head/cache 一致；重排前后混合 step 梯度相同；只有一次 optimizer/scheduler step；采样恢复一致。
- 工程：主干切权重不污染 student optimizer；head 不 alias 可变 actor 权重；预算和缓存引用可控；collective 顺序稳定。
- 性能：对比现有 sampled OPD、小规模 dense baseline、hidden+chunked 方案，固定有效 token 数、并行配置和测量口径，同时报告吞吐与峰值内存。
- 效果：分别评测各 teacher 对应领域及通用能力，观察能力保留与领域权重响应；全词表目标降低采样噪声不等于保证所有专家能力无损合并。

本文已完成源码路径和设计约束核对，以及本机 GPU 型号读取。尚未运行 GPU kernel、分布式训练、显存或 tokens/s 基准，因此没有给出实测速率或收益承诺。
