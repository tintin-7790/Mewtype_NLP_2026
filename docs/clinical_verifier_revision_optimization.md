# Mewtype D Pipeline：紧凑核验与局部修订优化

版本：`2026-11-24.clinical-v4.4-compact-verifier-v1`
范围：D Pipeline 的 Claim Verifier、Local Reviser、状态日志与回归测试。本文不宣称临床质量已经提高；真实 Qwen 复测仍待具备授权配置的环境执行。

## 1. 真实旧 D 故障结论

已只读核对 `clinical_run_log_d_retest.json`、`clinical_output_d_retest.json` 和对应代码。PRACTICE01 的旧 D 实际只完成 Planner 与 Writer，Verifier 没有完成 JSON 解析，Local Reviser 也没有被 Qwen 调用。

| 阶段 | Prompt Token | Completion Token | 总 Token | 耗时 | 结果 |
|---|---:|---:|---:|---:|---|
| Planner | 9,884 | 1,432 | 11,316 | 32,596.57 ms | 成功 |
| Writer | 15,596 | 981 | 16,577 | 22,967.02 ms | 成功 |
| Verifier | 8,263 | 2,048 | 10,311 | 45,867.89 ms | `INCOMPLETE_JSON` |
| 合计 | 33,743 | 4,461 | 38,204 | 101,446.11 ms | 3 次调用 |

根因是旧 Verifier 要为 28 条 Claim 逐条返回 `claim_id/verdict/rationale/source_ids`，再附加 issues；全局 `max_tokens=2048` 恰好被耗尽，JSON 在字符 3634 的字符串中截断。`json_schema` 只约束返回结构，不能保证内容在输出上限内结束，所以仍会产生不完整 JSON。当前 `max_tokens` 是 `ModelConfig` 的全局调用参数，不是分阶段参数。

旧请求还同时携带完整 sources、clinical plan、完整 draft、claims 及确定性提示，存在内容重复。旧输出中的 `revision_applied=true` 实际对应 `deterministic_conservative_fallback`，且把“自动核验未完成”写进了临床 `review_items`；两者均会造成误读。

## 2. Compact Claim-Level Verification

### 输入

Verifier 只接收：

- 当前病例截止点与原始 sources（患者事实边界）；
- 紧凑 Claim：`id/section/type/text/declared_source_ids`；
- 可选检索候选的来源编号、分数及兼容性警告；
- Python 确定性检查的紧凑提示。

不再重复传入完整 draft 和 clinical plan。Sources 仍保留，避免仅凭 Planner、检索或模型常识证明患者事实。

### 输出 Schema

```json
{
  "checked_claim_ids": ["C001", "C002"],
  "issues": [
    {
      "claim_id": "C002",
      "code": "UNSUPPORTED_NEGATIVE",
      "severity": "high",
      "hint": "原始资料未说明当前治疗状态",
      "source_ids": []
    }
  ]
}
```

正确 Claim 不输出 verdict 或 rationale。Python 强制 `checked_claim_ids` 无重复且与输入 Claim 集合完全相等；缺失、重复或未知 ID 返回 `INCOMPLETE_CLAIM_COVERAGE`，不会把模型沉默解释为通过。issue 的 Claim、错误码、风险级别和 source ID 均单独校验；硬性事实错误不能降为 medium/low。

错误码覆盖患者事实反转、数值、单位、时间、更正、未知写成否定、计划写成已完成、结果写成已取得、诊断确定性升级、推断依据不足、重要证据遗漏、计划关联不足、无引用推断、不安全建议和来源不匹配。Dense/Graph 相似度只辅助定位，不直接证明事实成立。

28-Claim Mock 回归中，完整无问题响应的序列化长度低于 500 字符，Schema 可解析且覆盖率为 100%。这只证明协议与程序行为，不代表真实 Qwen 一定在同样长度内稳定返回。

## 3. Claim Extraction 与 Localized Revision

正文 Claim 保存 `span_start/span_end`，evidence 和 review_items 保存 `item_index`。分句仍以句号、分号为稳定边界；仅对过长或含多个数值测量的分句进一步按逗号拆分，减少一个 Claim 混合多个可独立判断事实的情况。它不是医学级语义解析器，复杂并列和跨句指代仍需模型或人工审核。

Reviser 的响应改为 Patch：

```json
{
  "edits": [
    {
      "claim_id": "C003",
      "section": "plan",
      "action": "replace",
      "replacement": "当前抗感染治疗方案尚需核实。"
    }
  ],
  "review_items_to_add": []
}
```

约束如下：

1. 只允许编辑 Verifier 定位的问题 Claim；同一 Claim 的多个高风险 issue 合并为一个 Patch 目标，但原始 issue 日志全部保留。
2. section 必须与 Claim 定位一致；正文按字符区间倒序应用，列表按项目下标倒序应用，重复短语不会依赖全局替换。
3. 正文仅允许 replace；列表支持 replace/remove。未返回 Patch 的 issue 进入 `unresolved_claim_ids`。
4. Patch 后重新执行输出结构、来源编号及确定性高风险检查；仍有高风险时模型修订不记为成功。
5. 普通日志只保存 Claim ID、策略、幅度、哈希和状态，不复制病例全文。
6. JSON/API/模型解析等技术信息不得加入医生可见的 `review_items`。

若 Reviser 明确选择不修改且留下临床待核项，状态为 `completed_revision_unresolved`、`revision_applied=false`，对应 Claim 保留在 `unresolved_claim_ids`；不能把一次成功的 API 返回等同于成功修订。

模型修订失败或关闭时，可对已精确定位的 HIGH Claim 使用确定性保守替换；日志明确标为 `deterministic_conservative_fallback`，且不会设置模型含义的 `revision_applied=true`。

## 4. 状态和调用预算

Clinical 日志分别记录：

- `verification_enabled/attempted/completed/status/error`；
- `verified_claim_count/total_claim_count/verification_coverage/issue_count`；
- `revision_attempted/applied/strategy/revised_claim_ids/unresolved_claim_ids`；
- `deterministic_fallback_applied` 与 `revision_error`；
- 每阶段 model、response format、Token、耗时、JSON 解析结果、finish reason 和 max tokens。

若接口返回 `finish_reason=length|max_tokens`，或 completion tokens 等于配置上限，解析错误日志会标记 `output_truncation_suspected=true`。无法确认时不猜测。

通常调用数：D0/D1 无 HIGH 为 3 次，有 HIGH 且开启修订为 4 次；D2/D3 在 adaptive Agent 实际触发时各增加 1 次。Graph 检索不增加 Qwen 调用，但增加 Embedding 调用。程序沿用显式每例预算；不足时明确失败或记录未完成状态，不循环重试。

## 5. GraphRAG / Agent 兼容性

本次未重写 `clinical_graph.py`、`graph_retrieval.py`、`agent_orchestrator.py` 或 Counter-Evidence Prompt。Graph 仍只重排当前病例候选；Agent 仍最多一次且不修改患者 facts。已有 Mock 集成测试覆盖 Graph + Agent 控制流，新 D 代码共享同一 Writer 后 Claim 流程。

模式矩阵：

| 模式 | Graph | Agent | Qwen 阶段 |
|---|---|---|---|
| D0 | off | off | Planner, Writer, Verifier, 可选 Reviser |
| D1 | one_hop/ppr | off | 同 D0 |
| D2 | off | adaptive | Planner, 可选 Counter, Writer, Verifier, 可选 Reviser |
| D3 | one_hop/ppr | adaptive | 同 D2 |

## 6. 已测试、真实验证与未验证

### 已通过自动测试

- 28 Claim 紧凑响应、完整覆盖、错误来源、硬错误风险等级；
- 不完整/尾随/双 JSON 的明确解析失败；
- 未知写成否定、计划写成完成、数值、单位、更正和时间规则；
- 重复短语只修改指定 Claim；
- LOW 不触发 Reviser、MEDIUM 不阻断、HIGH 触发一次 Patch；
- 模型修订与确定性 fallback 的日志语义分离；
- Baseline、Evidence Pipeline、Planner/Writer、Dense Retrieval、Graph 和 Agent 回归。

完整结果：`107 tests passed`；`python3.11 -m compileall -q src main.py tools tests` 通过。

### 已有真实 Qwen 证据

旧 D 的 Planner/Writer 成功以及 Verifier 截断数据来自真实 `qwen-plus` 日志。它不能证明新 Compact Verifier 或新 Local Reviser 已在真实 Qwen 上成功。

### 尚未验证

- 新 Compact Schema 在 `qwen-plus` 与赛方内网 Qwen 上的真实稳定性、Token 和耗时；
- Qwen issue 判断的医学准确性及 Patch 的临床质量；
- PRACTICE01/PRACTICE02 的人工医学审核结果；
- 正式接口对 json_schema、finish_reason、seed、调用预算和分阶段输出上限的支持。

当前 Codex 进程未继承 `DASHSCOPE_API_KEY/QWEN_BASE_URL/QWEN_MODEL`，因此没有发起公网调用，也没有生成伪造的 fixed 结果。`information/converted/02_input.json` 已存在，但同样未做真实调用。

## 7. 真实复测命令

仅在本机已配置获准的百炼开发变量后执行一次：

```bash
python3.11 main.py \
  --input information/converted/01_input.json \
  --output clinical_output_d_fixed.json \
  --log clinical_run_log_d_fixed.json \
  --mode dev_api \
  --pipeline clinical_pipeline \
  --clinical-verification on \
  --clinical-revision on \
  --clinical-prompt-profile contrastive \
  --clinical-response-format json_schema
```

该命令不会覆盖旧 D 文件。复测时应确认：Verifier `verification_completed=true`、coverage=1、stage JSON 解析成功；只有出现 HIGH 且 Reviser Patch 成功时 `revision_applied=true`；否则查看 `deterministic_fallback_applied` 与 `unresolved_claim_ids`。随后逐条人工核对数值、单位、否定、时间、诊断确定程度、计划/已完成状态和 Patch 前后上下文。

PRACTICE02 可在 PRACTICE01 单例稳定后用同一命令替换输入与输出文件名运行一次，不应直接批量并发。
