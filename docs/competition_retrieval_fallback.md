# 正式比赛检索策略与降级建议

## 1. 当前默认不变

`src/competition_profile.py` 中的 `ACTIVE_COMPETITION_PROFILE` 仍为 `baseline_safe`。本轮只实现并离线验证检索策略，不会让正式入口自动切换到尚未完成真实 Qwen/BGE-M3 验收的 Clinical Profile。

正式 `.env` 仍只读取赛方提供的 `MODEL_NAME` 和完整 `ENDPOINT`。Embedding 的 HTTP 地址、认证、请求体、超时和配额未公布前，`OfficialEmbeddingClient` 不猜测协议，也不会访问公网。

## 2. 三种策略

| 策略 | 种子来源 | Graph | Embedding | 失败后的行为 |
|---|---|---|---|---|
| `dense_graph` | 赛方正式 BGE-M3 Dense | PPR | 仅 official | Dense 异常后尝试已离线验证的 `lexical_graph`，再失败则 `no_graph` |
| `lexical_graph` | 标准库实现的病例内 BM25 风格词法分数 | PPR | 无 | 词法或 Graph 异常后转 `no_graph` |
| `no_graph` | 无 | 关闭 | 无 | Planner、Agent、Writer、Verifier/Reviser 正常运行 |

词法检索复用现有病例内 `RetrievalUnit`、`source_id`、`field_path`、时间/否定/状态兼容提示和同一 Clinical Graph。它只替换 PPR 的初始种子，不修改 Fact、Source、Graph 边或最终 evidence。中文使用字符 unigram/bigram，英文和数值使用词元；不引入 jieba、BM25 第三方库、模型权重或网络服务。

相似度或词法分数均不能证明事实正确。Verifier/Reviser 始终回到原始 Sources，Agent 在没有检索候选时仍接收原始 Fact 目录、Source 目录和 Planner 问题。

## 3. 构建时 Profile 建议

项目提供三个尚未激活的完整 Profile：

- `clinical_d_graph_agent`：请求 `dense_graph`；只有赛方 BGE-M3 协议和预算真实验收后才考虑启用。
- `clinical_d_lexical_graph_agent`：直接使用 `lexical_graph`；不依赖 Embedding 服务。
- `clinical_d_no_graph_agent`：使用 `no_graph`；保留 Planner、Counter-Evidence Agent、Writer、Verifier/Reviser。

建议决策顺序：

1. 赛方提供 BGE-M3 后，先实现并验证 `OfficialEmbeddingClient`，固定病例测试维度、批量、超时、错误码、耗时和配额，再考虑 `clinical_d_graph_agent`。
2. 若没有 BGE-M3，优先在正式容器/Python 3.11 上复跑词法+PPR固定集；通过后可选择 `clinical_d_lexical_graph_agent`。
3. 若词法+PPR在正式容器未通过，选择 `clinical_d_no_graph_agent`。
4. 在真实 Qwen调用预算也未确认时，继续保持当前 `baseline_safe`，不要仅因离线测试通过就切换。

切换正式 Profile 是打包前代码常量变更，不增加赛方 `.env` 字段，也不根据本机百炼变量自动选择。任何切换都应重新执行全量测试、无参数 Mock HTTP 集成、固定病例人工审核和包白名单检查。

## 4. 审计字段

每例 `clinical.graph_retrieval` 记录：

- `retrieval_strategy_requested`：请求策略；
- `retrieval_strategy`：实际完成的策略；
- `retrieval_fallbacks`：每次降级的 from/to/reason；
- `embedding_provider`：尝试的 Dense Provider，无 Dense 时为 `null`；
- `seed_provider`：实际产生 Graph 种子的 Provider；
- `graph_requested`、`graph_executed`、`graph_fallback_reason`；
- Fact/Source 候选、PPR重排、边类型和耗时；
- `adaptive_agent.completed/error`：Agent 是否实际完成。

Graph 或 Embedding 失败不会自动把已解析病例判为失败，但日志不会把未执行的 Graph 标记为成功。结构错误、非法来源和未修复 HIGH 患者事实风险仍由原有硬约束处理。

## 5. 离线验证范围

离线测试覆盖：

- 模拟 official BGE 正常返回后执行 Dense+PPR；
- BGE 缺失和超时后转 Lexical+PPR；
- Lexical+PPR 成功；
- 词法检索异常后转 No-Graph；
- No-Graph 下 Agent 仍基于 Sources/Facts/Planner 工作；
- 三条路径 Qwen 调用数相同，没有为检索降级增加模型调用。

这些测试验证工程控制流，不证明词法检索、BGE-M3 或生成病历的医学质量。
