# MAA 设计决策记录

## 决策 1：记忆是无限的，物理容量是有限的

**日期**：2026-04-10

**问题**：`max_size` 应该被视为什么？是对记忆能力的限制，还是物理护栏？

**讨论**：

- 用户原话："我觉得记忆应该不限制大小，可以无限大小"
- 后续细化："记忆可以归纳和总结，max_size 可以保留，只是物理上的限制"

**结论**：

- 概念上：记忆是无限的，每个曾经见过的 token 都应该被记住。
- 物理上：`max_size` 只是一条护栏，防止 OOM。
- 主路径：归纳/总结（consolidation）把旧记忆压缩成摘要。
- 兜底路径：LRU 丢弃（仅在归纳未及时执行时触发）。

**实现**：README 中新增"记忆是无限的，物理容量是有限的"小节，`compress_by_time` 作为主动归纳工具。

---

## 决策 2：Y/N 作为普通 token，不修改 MemoryBank 架构

**日期**：2026-04-10

**问题**：用户要求"错误的记录也要给模型，让模型知道错误"，是否需要修改框架？

**选项**：

- A：修改 MemoryBank，给 entry 加 `verdict` 字段
- B：把 `Y`/`N` 作为普通 token 加入词表，不修改框架

**结论**：选 B。

**理由**：

- 实验证明 `Y`/`N` 作为普通 token 即可被模型学习（PrefixNoisy 100%）。
- 不修改框架 = 更通用，任何自回归任务都可以用这个机制。
- `verdict` 字段可以作为未来演进方向（多维度权重的一部分），但不是当前必需的。

---

## 决策 3：因果掩码作为框架级扩展

**日期**：2026-04-10

**问题**：`UnifiedMemoryAttention` 默认双向，训练时 teacher-forcing 导致注意力泄漏。

**症状**：eval_acc 100%，测试 Cold 25%。

**方案**：给 `forward` 添加 `causal: bool = False` 参数。

```python
if causal:
    cm = torch.zeros(T, T_all, device=device)
    seq_triu = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
    cm[:, T_all - T:] = seq_triu
    logits = logits + cm
```

**关键设计**：

- memory-bank 列始终可见（bank 代表历史，不是未来）。
- seq 部分做上三角因果掩码。
- 默认 `False` 保持向后兼容。

---

## 决策 4：停止信号监督

**日期**：2026-04-10

**问题**：eval_acc 1.0，但 greedy 推理时多生成 token（`0+4=` → `44`）。

**根因**：loss 只监督答案位置（`ans_start` 到 `ans_end`），答案后的 PAD 无监督。

**修复**：

```python
# 原代码
for i in range(t_ans_start, t_ans_end):
    mask[i] = 1.0

# 修复后
for i in range(t_ans_start, t_ans_end + 1):
    mask[i] = 1.0
```

把监督延伸到 `ans_end + 1`（即答案后的第一个 PAD），让模型学会"答案结束 → 输出 PAD"。

**验证**：修复后 L1 Cold 从 45.5% → 100%，L2 Cold 从 63% → 100%。

---

## 决策 5：从 TimeBias 到多维度动态权重

**日期**：2026-04-10

**问题**：Bank 模式始终低于 Prefix 模式（L2: 89.5% vs 100%）。

**根因分析**：

- demos 写进 bank 时时间戳从 0 开始。
- query 时 `current_step` 跳到 5（5 个 demo 之后）。
- TimeBias 惩罚了时间差：`log(1+5) ≈ 1.8`。
- **教学 demos 不应受时间惩罚**——它们的价值取决于正确性，而非新旧。

**结论**：时间只是记忆价值的一个维度。不同场景下，不同维度的权重不同。

**演进方向**：

```
MemoryBias(entry) = w_time·f_time + w_verdict·f_verdict + w_freq·f_freq + w_source·f_source
```

权重由当前 query 通过一个轻量 MLP **动态决定**。

| 记忆类型            | 主导维度      | 例子                 |
| ------------------- | ------------- | -------------------- |
| 情景型 (episodic)   | 时间 + 频率   | "用户昨天说喜欢蓝色" |
| 知识型 (semantic)   | 可信度 + 来源 | "1+1=2"              |
| 程序型 (procedural) | 频率          | "怎么骑自行车"       |

---

## 决策 6：训练时 `memory_bank=None`

**日期**：2026-04-10

**问题**：训练时跨 batch 共享 bank，模型 eval_acc 100% 但测试 25%。

**根因**：bank 变成答案查询表，模型学会走捷径。

**方案**：训练时 `memory_bank=None`，推理时才用 bank。

**代价**：模型在训练中从未接触过 bank，Bank 模式测试时会略差（89.5% vs 100%）。

**未来改进**：

- Bank 写入加随机 Dropout（50% 概率跳过）。
- Bank 里只存"通用知识"，不存具体样本。

---

## 待决策事项

1. **是否引入 `U`（Uncertain）token？**
   - 当两个 demos 冲突时（同一问题给出不同答案），模型应输出 `U` 表示无法判断。
   - 需要扩展词表 + 构造冲突训练数据。

2. **记忆类型是否显式标注？**
   - 方案 A：用户/调用方显式标注 `mem_type="semantic"`。
   - 方案 B：模型自动分类（加一个小的类型分类器）。

3. **跨层共享 bank 的梯度问题**
   - 如果所有层共享同一个 bank，梯度会通过 bank 反向传播到前面的层。
   - 需要设计一个"银行写入是 stop-gradient"的机制。

---

## 决策 7：DynamicMemoryBias 初始化策略

**问题**：DynamicMemoryBias 的 MLP 权重生成器如何初始化？

**决策**：最后一层偏置初始化为 [1, 0, 0, 0]（时间=1，其余=0），权重矩阵初始化为零。

**理由**：确保初始行为等同于纯 TimeBias，训练从已验证的基线起步，避免随机初始化导致的训练不稳定。

**验证**：训练 eval accuracy 达到 100%，无收敛问题。

---

## 决策 8：推理时 Padding 对齐

**问题**：推理时短序列的 query_context 与训练时长序列的分布不匹配。

**决策**：推理时将输入 pad 到 MAX_SEQ_LEN，从最后一个实际 token 位置读取预测。

**理由**：DynamicMemoryBias 使用 `x.mean(dim=1)` 计算 query_context，序列长度不同会改变均值语义。Pad 到训练长度确保分布一致。

**验证**：Cold L1 从 59.5% 恢复到 98.5%，Cold L2 恢复到 100%。

---

## 决策 9：记忆类型先验作为软偏置

**问题**：MemoryType 的先验权重（episodic 偏时间，semantic 偏可信度）如何参与计算？

**决策**：先验作为 MLP 初始化偏置的参考，但不硬编码。MLP 可以在训练中学习覆盖先验。

**理由**：不同任务的最优维度权重可能不同于先验假设。硬编码会限制模型灵活性。

**验证**：在算术任务中，模型成功学会了以 verdict 维度为主（教学场景）。
