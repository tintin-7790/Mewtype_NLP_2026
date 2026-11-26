# 北京语言大学Mewtype — Final Validation Recovery v1

日期：2026-11-25
范围：确定性日期/数值校验、Evidence 局部恢复、正文数值安全门和运行审计。未修改五阶段 Prompt、Graph、Agent、Shared Clinical Memory、正式输出 Schema 或比赛 Profile；未调用真实 Qwen。

## 1. PRACTICE05 实际失败

确认历史文件 `information/results/05_20261125_082754/output.json` 是 PRACTICE05 的失败输出：`status=failed`、`medical_record=null`、错误为 evidence[0] 的“2026年”和“10日”未在引用来源中找到。

同目录 `run_log.json` 的唯一病例是 `DEMO001`，不是 PRACTICE05，因此不能证明 PRACTICE05 的 Planner、Writer、Verifier 或 Reviser 实际状态。本轮没有用该日志推断模型阶段行为。失败输出也没有保存 Writer 初稿或局部 Evidence 内容，因此无法从现有历史文件恢复当时完整 medical_record。

原始 PRACTICE05 病历使用 U+2011 非标准连字符：`2026‑01‑10`。适配后相关来源包括：P0008 入院时间、P0009 记录时间、P0014 现病史和 P0036 辅助检查时间。旧校验先执行日期正则，再做 Unicode 规范化，导致带 U+2011 的来源日期没有进入等价日期格式；随后通用数字正则把生成文本中的中文日期拆成“2026年”“10日”。这是日期格式误报，不是已经证明的患者时间冲突。

## 2. 日期和数值校验

`normalize_equivalent_dates` 现在先执行 NFKC、μ/µ 和 Unicode 连字符规范化，再把以下完整日期转换为严格的 `YYYY-MM-DD`：

- `2026-01-10`
- `2026/01/10`
- `2026年1月10日`
- `2026年01月10日`

完整日期先独立比较，并在通用数字检查前被遮蔽，因此不再拆成年、月、日三个数字。`2026-01-10` 与 `2026-01-11` 仍触发 `DATE_SUPPORT`；来源没有日期时也不会通过。

检验数值和单位继续精确比较。正文中的日期及带单位数值增加 HARD 安全门：原始 Sources 不支持的 `128g/L`、错误单位或错误患者日期不能通过删除 Evidence 规避。

## 3. Evidence 局部恢复

新增 `recover_evidence`，不修改三段正文，只处理 Evidence 数组：

1. 记录初始 Evidence 错误和警告。
2. 对日期执行严格等价归一化。
3. 来源编号错误时，仅允许将 fact Evidence 重新绑定到唯一一个“规范化原文直接包含陈述”且通过日期、数值、否定和状态检查的 Source；不使用 Dense、Graph、相似度或医学常识猜测。
4. 对合法短来源中的 fact 数值/日期错误，可复用原有 exact cited source 方式重建该条 Evidence。
5. `PLAN_AS_DONE`、无唯一支持来源、推断中的无依据数值等无法安全恢复项仅从 Evidence 数组删除。
6. 重新执行全部 Evidence 和正文安全检查。合法 Evidence 少于三条时按实际数量输出，不填充来源。

删除 Evidence 不会删除或改写 case_features、diagnostic_discussion、plan，也不会向 review_items 添加技术性或无对象模板。若同一个不支持数值仍存在于正文，`UNSUPPORTED_BODY_NUMERIC` 会继续阻断整例。

审计字段位于病例运行日志的 `final_validation_recovery`，Clinical Writer 和 post-revision 阶段也在各自 evidence validation 中保留恢复报告：

- `evidence_validation_initial`
- `evidence_repairs`
- `evidence_dropped`
- `evidence_validation_final`
- `source_reassignment`
- `date_normalizations`
- `residual_evidence_risks`
- `recovered_success`
- `partial_evidence_recovery`
- `additional_model_calls`（固定为 0）

日志只保存索引、错误码、来源编号和陈述哈希，不复制完整患者 Evidence。

## 4. 风险分级执行

### HARD

最终 Schema/正文缺失、正文无来源日期或数值、括号残缺、通用审稿模板，以及其他无法安全消除的确定性高风险仍阻断成功。Claim Verifier 的患者事实反转、计划写成已完成、待排升级确诊、内部 ID 泄漏等原有 HIGH 规则保持不变。

### REPAIRABLE

Evidence 结构、来源编号、日期格式、Evidence 数值、计划状态和不可操作 review item 优先进行确定性局部恢复。恢复失败时只删除该 Evidence；若风险已进入正文，仍由 HARD/Claim 门处理。

### SOFT

字面支持有限、合理但证据有限的推断等 warning 保留审计，不直接使完整病例失败。医学不确定性本身不是生成错误。

## 5. 安全输出条件

只有最终记录满足原输出 Schema、三段正文非空、所有保留 Evidence 有合法来源、没有不支持的正文数值/日期、没有残余边界错误时才返回 success。恢复状态只写入内部日志，不修改 `status` 枚举或 medical_record Schema。

本轮没有加入宽泛的“安全摘要重建”。在无法确定性保留临床含义时，宁可保留失败和明确风险，也不通过模板或规则生成新的诊断、治疗建议。`safe_content_reconstruction=false` 会写入审计。

## 6. 修改范围

- `src/validation.py`：完整日期识别、Evidence 恢复、正文日期/数值安全门和风险分级。
- `src/pipeline.py`：Writer、post-revision 和最终输出前接入恢复及审计，不增加模型调用。
- `src/evidence_policy.py`：无效 Evidence 仅记日志，不再向 review_items 写通用技术提示。
- `tests/test_final_validation_recovery.py`：新增21项定向回归。
- `tests/test_evidence_policy.py`、`tests/test_agent_orchestrator.py`、`tests/test_system.py`：更新新恢复策略下的旧断言。
- `tools/build_package.py`：登记本报告。

## 7. 验收结果

- PRACTICE05 原始 JSON 自动适配：PRACTICE05，49个来源，截止点 `2026-01-10 14:10`。
- PRACTICE05 静态日期回归：中文日期与 P0014 中 U+2011 日期等价，不产生 DATE_SUPPORT/NUMERIC_SUPPORT 误报。
- 定向测试：77/77 通过。
- 全量 unittest：228/228 通过。
- 新增 Qwen 调用：0。
- 正式输出 Schema、五阶段 Prompt、Profile：未修改。

## 8. 仍会合理失败的情况

整例仍会在以下情况下失败：模型 JSON/必需字段无法恢复，三段正文为空，正文含 Sources 未支持的患者日期或带单位数值，存在无法安全消除的患者事实反转/确诊升级/计划状态错误，正文结构损坏，或仍泄漏内部审计信息。这些失败是安全边界，不应为了提高成功率放行。

正式打包仍需一次隔离 ZIP 复现和赛方 Qwen Endpoint 联调。当前正式 Profile 继续为 `baseline_safe`；赛方 BGE-M3 契约与完整 Clinical Profile 的正式预算仍待确认。
