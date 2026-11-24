# v0.4.3 生成算法优化与实验状态

日期：2026-11-24。本文只记录已实现和已执行内容，不声明未经医学审核的质量提升。

## 已实现

- Contrastive Clinical Planner：复用既有 Schema，要求每个问题显式关联支持事实、明确限制、少量鉴别、待区分资料和 plan goals；Python 去重并将鉴别方向上限设为3。
- Problem-Conditioned Writer：Python 将 Fact ID 解析成 `problem_evidence_context`，按问题提供支持、限制、鉴别、待区分资料、plan goals 和按问题归组的检索候选。
- Evidence Salience Ranking：对问题关联、数值、明确阴性、更正、时间、状态、异常检查、暴露及治疗史加权，对无关人口学字段降权；只排序，不改写事实。
- Dense 查询增强：在 Planner 查询后补入当前问题实际引用的患者事实、限制和待区分资料；仍只检索当前病例，且相似度不作为事实证明。
- Prompt Profile：`legacy`、默认 `contrastive`、每阶段最多一个合成示例的 `contrastive_fewshot`。
- 非阻断核验：Verifier JSON/verdict 失败时保留合法 Writer 草稿并记录核验未完成；Reviser 失败时进行已定位事实的保守回退。
- Planner 冲突规范化：若同一鉴别诊断的支持/反对列表包含同一 Fact ID，先验证 Fact ID 确实存在，再从两个方向同时移除该歧义引用，并在 `missing_evidence` 与运行日志中标记“方向需复核”；原始 facts 与模型原始 plan 不被改写。
- 有限降级：Contrastive Planner 规范化后仍无效时，只在剩余预算允许时尝试一次 Legacy Planner；否则可用最后一次调用执行 Baseline。日志以 `algorithm_mode` 与 `planner_fallback` 如实记录实际路径，预算不足则明确失败。
- 上下文去重：Planner 不再随 facts 重复发送 `raw_text`；Writer 只接收一份原始 sources 与一份 Fact-ID catalog，问题/鉴别关系仅引用 Fact ID。每阶段日志记录系统提示词、用户载荷和总输入字符数，但不记录病例正文。

## 已执行验证

- 自动化测试、Python 编译和 diff 检查见本次开发报告。
- 对可解析的 PRACTICE01–06 执行了不涉及模型和截止点推断的静态显著性检查：每例 Top-10 事实中的无关人口学字段均为0。
- PRACTICE07 原始文件存在尾随额外 JSON 内容；PRACTICE08 原始文件缺少 JSON 分隔符。原件未修改。

## PRACTICE01 已有真实日志审计

本轮只读取 `information/results/v043` 下既有 A/B/C/D 日志，没有新增云端调用。阶段数据如下：

| 组别 | 阶段 | Prompt | Completion | Total | 耗时 ms |
|---|---|---:|---:|---:|---:|
| A | model | 3549 | 1670 | 5219 | 37411.51 |
| B | planner | 13078 | 881 | 13959 | 20563.49 |
| B | writer | 22886 | 923 | 23809 | 22238.81 |
| C | planner | 13438 | 1383 | 14821 | 31614.76 |
| D | planner | 13438 | 1098 | 14536 | 25282.98 |

B 合计 prompt/completion/total 为 35964/1804/37768。C、D 在 Planner 校验处终止，Writer 未运行。旧日志没有请求字符统计或完整 Planner 原始响应，不能从日志精确反推 Writer 各副本的字符占用；新版会用非敏感长度字段补齐这一审计能力。

对同一 PRACTICE01 输入的确定性载荷审计显示：50 个 sources、56 个 facts；旧 Planner 用户载荷 23794 字符，新精简载荷 18722 字符，减少 5072 字符（21.32%）。旧 facts 中 `raw_text` 累计 4144 字符，而唯一 source content 为 1955 字符，确认存在重复展开。此结果只说明请求载荷减少，不等同于实际 Token 或医学质量必然改善，需重新调用相同模型验证。

## 尚未执行的质量实验

当前进程未设置 `DASHSCOPE_API_KEY/QWEN_BASE_URL/QWEN_MODEL`，也没有可直接使用的本地 BGE-M3 配置；因此未执行新的真实 Qwen A/B/C/D 或 BGE-M3 E 组实验。8例完整实验还需要：

1. 修复后的 PRACTICE07/08 副本；
2. 团队显式确认每例 `record_cutoff`，不能自动把“记录时间”当作截止点；
3. 真实 Qwen 配置；
4. E组所需的已获准 BGE-M3 配置；
5. 独立人工标注，用于关键事实错误、诊断依据覆盖、限制依据准确性和计划对应率。

在这些条件满足前，不能宣称 Contrastive Planning、Few-shot 或 Dense Retrieval 已提高临床质量。
