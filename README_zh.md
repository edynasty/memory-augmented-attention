# 记忆增强注意力（Memory-Augmented Attention）

基于 PyTorch 的**统一记忆增强注意力（MAA）**实现。当前序列的 token（短期记忆）与历史 KV 对（长期记忆）统一存储在带时间戳的共享**记忆库（Memory Bank）**中，并通过**时间衰减偏置**进行联合注意力计算。

![MAA 架构图](MAA.png)

---

## 背景与动机

### 标准 Transformer 的局限

标准 Transformer 是强大的预测器，但本质上是**无状态的**：

- 每次前向传播只能看到**当前上下文窗口**中的内容。
- 推理时模型**权重固定**——跨调用不会"记住"任何东西。
- 唯一的"记忆"是输入的 prompt，受 `max_seq_len` 限制。

因此，标准 Transformer 计算的本质是：

```
y_t = P(token | 当前上下文)
```

它是一个纯函数，没有持久状态。

### 核心洞察：序列 Token 本身就是记忆

本库的核心出发点是：**当前序列的 token 本身就是一种短期记忆**。传统记忆增强 Transformer 将它们视为两类不同的东西：

```
Attention(Q, [K_seq ; K_memory], [V_seq ; V_memory])   # 两个独立的池子
```

但这种区分并没有本质依据。每个 token——无论来自当前输入还是过去的交互——都只是一个在特定时刻存储的 `(Key, Value, Query, 时间戳)` 元组。

本库将两者统一在同一个抽象下：

```
M_all = M_memory ∪ { (K_seq, V_seq, Q_seq, t_seq) }
y_t   = Attention(Q_t, K_all, V_all, TimeBias(t_t, t_all))
```

时间成为每条记忆的显式维度，注意力机制自动对时间上更久远的条目降权。

### 为什么时间维度至关重要

没有时间的记忆是无序的。为每条记忆条目加上时间戳 `t` 后，模型可以：

1. **偏好近期上下文** —— 通过时间衰减偏置实现。
2. **理解事件顺序** —— 模型可以学习 `t_i < t_j` 意味着事件 `i` 先于事件 `j` 发生。
3. **实现遗忘机制** —— 旧的、相关性低的条目自动被折扣，无需显式删除。

### 实验验证：算术教学任务

我们在字符级加法任务上验证了 MAA 的核心假设——**统一记忆可以让模型区分可信与不可信信息**：

- **训练**：只教 0–99 的混合位数加法，每个样本附带 0–6 条 in-context 演示（其中约 30% 标为错误 `N`）。
- **测试**：5 级难度（1 位 → 4 位 → 3 操作数），4 种条件：

| 条件            | 说明                                  | L2（训练内分布） |
| --------------- | ------------------------------------- | ---------------- |
| **Cold**        | 无演示，纯靠权重                      | **100%**         |
| **Prefix**      | 5 条正确演示拼在 prompt 前            | **100%**         |
| **PrefixNoisy** | 3 正确 + 2 错误演示（标 `N`）         | **100%**         |
| **Bank**        | 演示先写进 MemoryBank，query 单独输入 | **99.5%**        |
| **BankDynamic** | 混合演示带可信度元数据写入 Bank       | **100%**         |

关键结论：

- **Cold 100%** 证明模型从权重中学到了加法规则，不是简单记忆。
- **PrefixNoisy 100%** 证明模型能读取 `Y/N` 判决，**不盲抄错误演示**。
- **Bank 99.5% ≈ Prefix 100%** 证明 demos 写进 bank 后仍能被有效检索，统一记忆假设成立。
- **BankDynamic 100%** 证明 DynamicMemoryBias 能在可信度元数据可用时正确提升可信记忆权重、抑制错误记忆。
- **L3/L4（3-4 位）≈ 0%**：字符级小模型无法泛化进位链，这是模型容量/数据分布问题，非框架缺陷。

---

## 核心公式

### 统一记忆注意力

```
M_all = M_memory ∪ { (K_seq, V_seq, Q_seq, t_seq) }

y_t = softmax( (Q_t · K_all^T) / sqrt(d_k) + TimeBias(t_t, t_all) ) · V_all
```

### 时间衰减偏置

```
TimeBias(t_q, t_k) = -alpha * log(1 + |t_q - t_k|)
```

`alpha` 是一个**可学习标量**。对数形式与人类记忆中观察到的幂律遗忘曲线吻合：近期条目受到的惩罚较小，而非常久远的条目受到更大惩罚，但惩罚差距在时间跨度极大时趋于收窄。

令 `alpha = 0` 即退化为**标准（不感知时间的）注意力**。

### 从单一时间到多维度动态权重（设计演进方向）

实验表明，**时间只是记忆价值的一个维度**。在算术教学中，demo 的"正确性"（`Y`/`N`）比"新旧"更关键；在对话场景中，"使用频率"可能比"时间"更重要。

因此 MAA 正从"单一 TimeBias"演进为**多维度动态权重**：

```
MemoryBias(entry) = w_time·f_time + w_verdict·f_verdict + w_freq·f_freq + w_source·f_source
```

其中权重 `[w_time, w_verdict, w_freq, w_source]` 由当前 query 通过一个轻量 MLP **动态决定**——不同问题自动激活不同维度的记忆筛选逻辑。

| 维度                   | 函数                    | 典型场景权重             |
| ---------------------- | ----------------------- | ------------------------ |
| **时间** `f_time`      | `exp(-|Δt|/τ)`          | 对话历史高，数学教学低 |
| **可信度** `f_verdict` | `[-1, +1]`（`N`→`Y`）   | 教学演示高，闲聊低       |
| **频率** `f_freq`      | `log(1 + access_count)` | 代码补全高，一次性查询低 |
| **来源** `f_source`    | `[0, 1]` 权威性         | 知识问答高，匿名论坛低   |

### 并非所有输入都应被记忆

重要性过滤器防止低信息量的 token 污染记忆库：

```
write(entry)  仅当  ||V||_2 >= importance_threshold
```

---

## 复杂度与可扩展性

### 记忆是无限的，物理容量是有限的

从**设计语义**上讲，MAA 的记忆库是**无限**的：每一个曾经见过的 token 都应该被记住，时间维度上没有任何硬性截断。然而现实中的显存/内存是有限的，`max_size` 只是一条**物理护栏**，并非对"可记忆内容"的概念限制。

当记忆库触及物理上限时，MAA 通过**归纳与总结（consolidation）**把多条旧记忆压缩成少量摘要条目，从而在有限空间内保留更长时间跨度的信息 —— 类似人类把琐碎事件整合成长期记忆的过程。简单的 LRU 丢弃仅作为退化兜底。

### 注意力复杂度

对整个记忆库做全量注意力的复杂度为 O((L + M)² · d)，随着 `M` 增长会爆炸。MAA 提供三种互补策略将复杂度控制在可用范围内：

| 策略           | 复杂度       | 说明                                                                                      |
| -------------- | ------------ | ----------------------------------------------------------------------------------------- |
| **Top-K 检索** | O(L · K · d) | 每次只让最相关的 K 条参与注意力；K 为固定超参数（如 32–128）                              |
| **归纳/总结**  | O(L · d)     | 把旧条目压缩为摘要向量，保留语义、释放物理容量；调用 `bank.compress_by_time(keep_newest)` |
| **LRU 兜底**   | O(1)         | 物理上限触顶且未及时归纳时，丢弃最旧条目作为最后防线                                      |

推荐策略：

```
近期 token（< N 步）   → 全量注意力
较旧 token             → 仅 Top-K 检索
超出物理上限           → 归纳/总结成摘要向量（而非直接丢弃）
```

---

## 安装

```bash
pip install -e ".[dev]"   # 可编辑安装，含测试依赖
```

要求 Python ≥ 3.9，PyTorch ≥ 2.0。

---

## 快速上手

```python
import torch
from memory_augmented_attention import MemoryBank, MemoryAugmentedTransformerLayer

layer = MemoryAugmentedTransformerLayer(
    d_model=256,
    n_heads=8,
    d_ff=1024,
    dropout=0.1,
    top_k=64,        # 每次前向传播最多关注 64 条历史记忆
    init_alpha=1.0,  # 时间衰减强度初始值（可学习）
)
bank = MemoryBank(max_size=4096, importance_threshold=0.0)

global_step = 0
for batch in data_loader:
    x = batch["embeddings"]   # (B, T, 256)

    out = layer(
        x,
        memory_bank=bank,
        current_step=global_step,
        write_to_memory=True,  # 注意力计算后将当前序列写入记忆库
    )

    global_step += x.shape[1]   # 时间戳按序列长度递增
```

---

## API 参考

### `MemoryBank`

```python
MemoryBank(max_size=1024, importance_threshold=0.0)
```

每个槽位存储一个 token 的 `(key, value, query, timestamp)`。

> **关于 `max_size`**：概念上记忆库无上限，`max_size` 仅用作**物理容量护栏**以防 OOM。理想的使用方式是在逼近上限前主动调用 `compress_by_time` 将旧条目归纳成摘要，而不是依赖 LRU 默默丢弃。若你的场景内存充裕，可以把 `max_size` 调得足够大（例如 `10**7`），让它在实际运行中不触发淘汰。

| 方法                                                     | 返回值              | 说明                                             |
| -------------------------------------------------------- | ------------------- | ------------------------------------------------ |
| `write(key, value, query, timestamp)`                    | `bool`              | 写入一条记忆；若被重要性过滤器拒绝则返回 `False` |
| `write_sequence(keys, values, queries, start_timestamp)` | `int`               | 批量写入序列；返回下一个可用时间戳               |
| `retrieve_all(device=None)`                              | `(K, V, Q, T)` 张量 | 以堆叠张量形式返回所有条目                       |
| `top_k_retrieve(query, k, device=None)`                  | `(K, V, Q, T)` 张量 | 按余弦相似度返回最相关的 Top-K 条目              |
| `compress_by_time(keep_newest)`                          | —                   | 丢弃旧条目，只保留最新的 `keep_newest` 条        |
| `clear()`                                                | —                   | 清空所有条目                                     |

### `TimeBias`

```python
TimeBias(init_alpha=1.0)
```

可学习的时间衰减模块，仅含一个参数 `alpha`。
`alpha = 0` → 无衰减（退化为标准注意力）。

### `UnifiedMemoryAttention`

```python
UnifiedMemoryAttention(
    d_model, n_heads,
    dropout=0.0,
    top_k=None,       # None → 使用所有已存条目
    init_alpha=1.0,
)
```

基于统一记忆池的多头注意力。每次前向传播后，当前序列的 K/V/Q 投影会被写入 `MemoryBank`（当 `write_to_memory=True` 时）。

### `MemoryAugmentedTransformerLayer`

```python
MemoryAugmentedTransformerLayer(
    d_model, n_heads,
    d_ff=None,         # 默认为 4 * d_model
    dropout=0.1,
    top_k=None,
    init_alpha=1.0,
)
```

完整编码器层：`UnifiedMemoryAttention → Add & Norm → FFN → Add & Norm`。

---

## 记忆管理详解

记忆库在概念上是无限的，下列机制用于在有限的物理内存下尽可能无损地容纳它：

| 机制             | 配置参数                        | 行为                                             |
| ---------------- | ------------------------------- | ------------------------------------------------ |
| **物理容量护栏** | `max_size`                      | 仅用于防止 OOM；达到上限时触发 LRU 兜底          |
| **重要性过滤**   | `importance_threshold`          | 写入时丢弃 `‖V‖₂ < threshold` 的低信息条目       |
| **可信度标记**   | `verdict`（每条目）             | `+1.0`=确认正确，`-1.0`=确认错误，`0.0`=未验证   |
| **归纳/总结**    | `compress_by_time(keep_newest)` | 主动把旧条目压缩成摘要，在不丢信息的前提下瘦身库 |
| **Top-K 检索**   | 注意力层的 `top_k`              | 将注意力池限制为最相关的 K 条，降低计算复杂度    |
| **LRU 兜底**     | （自动）                        | 仅在归纳未及时执行、物理上限触顶时丢弃最旧条目   |

### 记忆类型与维度权重

不同类型记忆对维度敏感度不同：

| 记忆类型                | 例子                 | 主导维度      | 说明                         |
| ----------------------- | -------------------- | ------------- | ---------------------------- |
| **情景型** (episodic)   | "用户昨天说喜欢蓝色" | 时间 + 频率   | 旧信息价值下降，常用信息保留 |
| **知识型** (semantic)   | "1+1=2"              | 可信度 + 来源 | 正确性不因时间改变           |
| **程序型** (procedural) | "怎么骑自行车"       | 频率          | 熟能生巧，使用次数决定熟练度 |

---

## 训练指南

MAA 层可直接替换标准 `nn.TransformerEncoderLayer`，唯一额外需要关注的是跨步骤管理 `MemoryBank`。

### 有监督序列微调

```python
import torch
import torch.nn as nn
from memory_augmented_attention import MemoryBank, MemoryAugmentedTransformerLayer

model = MemoryAugmentedTransformerLayer(d_model=512, n_heads=8, top_k=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    bank = MemoryBank(max_size=2048)   # 每个 epoch 重置记忆库
    global_step = 0

    for x, y in train_loader:          # x: (B, T, 512)
        optimizer.zero_grad()

        out = model(x, memory_bank=bank, current_step=global_step)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        global_step += x.shape[1]

        # 可选：定期压缩防止记忆库无限增长
        if global_step % 4096 == 0:
            bank.compress_by_time(keep_newest=512)
```

### 关键超参数

| 参数                   | 推荐范围     | 作用                               |
| ---------------------- | ------------ | ---------------------------------- |
| `top_k`                | 32 – 128     | 越大 → 上下文越丰富，计算量越高    |
| `init_alpha`           | 0.5 – 2.0    | 越大 → 时间衰减越强                |
| `max_size`             | 1024 – 16384 | 内存中保存的最大历史条目数         |
| `importance_threshold` | 0.0 – 0.5    | 越高 → 只有显著 token 被存入记忆库 |

### 持久化记忆推理

```python
# 记忆库跨对话轮次持久存在
bank = MemoryBank(max_size=8192, importance_threshold=0.1)
global_step = 0

def respond(user_input_embedding):
    global global_step
    out = model(user_input_embedding, memory_bank=bank,
                current_step=global_step, write_to_memory=True)
    global_step += user_input_embedding.shape[1]
    return out
```

---

## 相关工作

MAA 处于多个活跃研究方向的交叉点：

| 论文 / 系统                               | 与 MAA 的关联                                                                            |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Memory-Augmented Transformers**（综述） | 系统整理了显式/隐式记忆、读写机制和容量管理，直接启发了本设计                            |
| **∞-former**（Martins et al., 2022）      | 通过连续空间注意力和"粘性记忆"实现无界长期记忆；与 MAA 的无限上下文目标一致              |
| **Memformer**（Wu et al., 2022）          | 用外部动态记忆模块编码和检索历史信息，含记忆回放反向传播；与 MAA 的写入-检索循环高度相关 |
| **Recurrent Memory Transformer（RMT）**   | 将记忆 token 注入输入/输出序列，在片段间传递状态；等价于将记忆条目视为一等公民序列元素   |
| **Memory Transformer**                    | 在输入前添加可学习全局记忆 token，捕获长程依赖                                           |
| **Longformer / BigBird**                  | 稀疏注意力模式，将 O(N²) 复杂度降低——与 MAA 的 Top-K 检索策略互补                        |
| **Neural Turing Machine / DNC**           | 可微分读写记忆；MAA 采用相同思路，但用软 Top-K 注意力代替独立寻址机制                    |
| **RAG**（Lewis et al., 2020）             | 检索增强生成：先检索后注意；MAA 将检索集成在注意力层内部，而非 prompt 层面               |

---

## 项目结构

```
memory-augmented-attention/
├── src/memory_augmented_attention/
│   ├── __init__.py          # 公开 API
│   ├── memory_bank.py       # MemoryBank + MemoryEntry
│   ├── attention.py         # TimeBias + UnifiedMemoryAttention + CausalMask
│   └── model.py             # MemoryAugmentedTransformerLayer
├── tests/
│   ├── test_memory_bank.py
│   └── test_attention.py
├── train_arithmetic.py      # 算术教学训练脚本（in-context + Y/N 判决）
├── demo_arithmetic.py       # 分级评测脚本（Cold/Prefix/Noisy/Bank × L1-L5）
├── arithmetic_maa_best.pt   # 训练好的 checkpoint（1.4M 参数）
├── MAA.png                  # 架构图
├── pyproject.toml
├── README.md                # 英文文档
└── README_zh.md             # 中文文档（本文件）
```

---

## 路线图

### 已完成

- [x] **因果掩码支持** —— 支持自回归训练/推理（`causal=True`）
- [x] **教学-测试分离验证** —— 通过算术任务验证统一记忆、Y/N 判决、Bank/Prefix 双通道
- [x] **停止信号监督** —— 让模型学会在答案结束后输出 PAD，解决 greedy 生成溢出
- [x] **多维度动态权重** —— DynamicMemoryBias：可学习的 [时间, 可信度, 频率, 来源] 动态加权，Bank L2 从 89.5% 提升至 99.5%
- [x] **记忆类型系统** —— MemoryType 枚举（Episodic/Semantic/Procedural）及维度先验
- [x] **可学习的归纳/总结** —— ConsolidationNetwork：交叉注意力压缩网络，256→88 条（~3:1）

### 进行中

- [ ] **分层记忆** —— 将记忆库按时效分为热/温/冷三层，分配不同检索预算

### 计划中

- [ ] **可学习重要性评分** —— 用小型 MLP 替代 L2 范数过滤器，预测 token 是否值得存储
- [ ] **跨层共享记忆库** —— 多层 Transformer 的所有层共享同一个 `MemoryBank`
- [ ] **稀疏注意力集成** —— 结合 Top-K 检索与 Longformer 风格的局部窗口注意力
- [ ] **长对话基准** —— 在 100+ 轮对话中验证 bank 的记忆持久性与时间衰减行为
- [ ] **不确定性 token** —— 词表扩展 `U`（Uncertain），让模型在记忆冲突时表达"无法判断"

---

## 运行测试

```bash
pytest tests/ -v
```
