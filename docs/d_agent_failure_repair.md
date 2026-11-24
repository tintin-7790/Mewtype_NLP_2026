# Mewtype D + Counter-Evidence Agent 故障修复报告

版本：v0.4.4-d-agent-repair-v1
日期：2026-11-24

## 1. 范围与结论

本轮只修复 D Pipeline 与 Counter-Evidence Agent 的真实运行故障，并保留现有 Compact Verifier、Claim Patch Reviser、Graph Retrieval 和 Adaptive Agent。未修改比赛输入输出协议，未新增依赖，未自动提交或推送代码。

旧运行中存在四类相互独立的问题：

1. Counter-Evidence Agent 输出被截断。旧日志显示该阶段 `completion_tokens=2048`、`finish_reason=length`，随后因不完整 JSON 失败。该问题不是最终 `VALIDATION_FAILED` 的直接原因，因为后续 Writer、Verifier 和 Reviser 仍继续执行。
2. 最终 evidence 中出现来源不支持的 `10μmol/L`，被 Python 数值—单位硬校验正确拒绝。旧日志未保存分阶段响应内容，因此无法可靠判定该数字最早由 Writer 还是 Reviser 引入；新版本增加分阶段 evidence 校验摘要，以便后续真实复测定位。
3. Verifier 把原始资料中明确记载的“无右上腹疼痛”误判为无依据否定，导致无意义修订及保守回退。根因同时涉及模型判断偏差和 Python 否定范围识别不足。
4. 最终校验失败时，临床阶段元数据尚未完整写入失败记录，导致失败病例缺少 agent、verifier、reviser、耗时及最终检查信息。

## 2. 旧运行证据

### D-only

- 状态：成功。
- Qwen 调用：4 次。
- 总 token：35,161。
- 总耗时：约 61 秒。
- 分阶段 token：Planner 10,491 + 1,289；Writer 14,773 + 904；Verifier 6,077 + 288；Reviser 1,283 + 56。
- Verifier 覆盖率为 1，但错误地质疑“无右上腹疼痛”；原资料 P0014 明确支持该否定事实。

### D + Counter-Evidence Agent

- 状态：失败，最终错误为 `VALIDATION_FAILED`。
- Qwen 调用：5 次。
- 总 token：49,834。
- 总耗时：约 136.5 秒。
- Agent 阶段：prompt 9,246，completion 2,048，`finish_reason=length`，随后发生 `INCOMPLETE_JSON`。
- 最终 evidence 使用 `10μmol/L`，但引用来源只包含 14.2 和 11.8 等病例实测值，未支持 10。

以上统计来自既有运行日志，不代表修复后性能，也不外推赛事内网模型表现。

## 3. 修复设计

### 3.1 Compact Counter-Evidence Agent

Agent 现在只接收：

- 来源目录（`source_id` 与 `field_path`，不重复整段原文）；
- 去重后的简洁事实及其限定信息；
- 问题编号、摘要、鉴别诊断和相关 Fact ID。

输出改为紧凑结构：

- `reviewed_problem_ids`：必须完整且仅覆盖当前全部问题；
- `issues`：只报告高价值反证、方向歧义、缺少关键鉴别证据或确定性过强。

Agent 仍最多调用一次。输出截断、JSON 不完整或问题覆盖不完整时，会记录明确错误并继续 Writer，不会伪装为 Agent 成功，也不会无限重试。

### 3.2 分阶段可审计性

每次模型调用日志记录：

- stage、model、response_format、schema；
- HTTP/解析/Schema 校验结果；
- prompt、completion、total token；
- 耗时、finish_reason、max_tokens；
- 是否截断及错误类型。

普通日志不保存完整病例或完整模型响应。Writer 后和最终输出前分别记录 evidence 校验摘要，使下次真实运行可以定位不支持的数值首次出现在哪个阶段。

### 3.3 Evidence 数值与单位保护

数值校验改为按“数值 + 单位 + 引用来源”配对，支持 `µ/μ`、Unicode 空格和连字符的规范化，但不再用“来源中出现过同一个裸数字”作为支持依据。

自动修复仅在以下条件同时满足时进行：

- evidence 类型为 `fact`；
- 引用编号真实存在；
- 引用的短原文能够被原样、安全地作为 statement；
- 不需要猜测指标、时间或附近数值。

医学知识阈值可以谨慎出现在拟诊讨论中，但不能伪装成患者实测值或 fact evidence。无法安全修复时继续由最终硬校验拒绝，不删除校验规则。

### 3.4 保守 Verifier 与局部修订

Python 侧增加显式来源保护：若原始来源明确包含同一时间语境下的否定事实，则模型产生的 `UNSUPPORTED_NEGATIVE` 或患者事实矛盾 issue 会被撤销并留下审计记录。

同时加强以下硬检查：

- 未提及不等于否认；
- 阳性事实不能被生成文本反转为否定；
- “未检查”不能被写成“结果阴性”；
- 同一症状在不同时间点的阳性与阴性可并存；
- 修订后必须重新执行结构、引用、数值、否定、时序和状态校验。

Reviser 仍最多额外调用一次，并只允许 Claim Patch。修订失败会保留 `patch_applied`、剩余风险和校验错误，不再丢失失败上下文。

### 3.5 失败日志完整性

临床 pipeline 的完整元数据现在先于最终校验构建。即使最终失败，输出日志仍保留：

- Agent 是否触发、是否完成及失败原因；
- Verifier 覆盖与来源保护结果；
- Reviser 是否触发及修订结果；
- 各阶段调用、token 和耗时；
- Writer 后与最终 evidence 校验摘要；
- `failed_stage` 和结构化 validation errors。

## 4. 自动化验证

新增或补强的测试覆盖：

- Agent 紧凑输入不重复完整来源；
- 问题覆盖不完整时明确拒绝；
- `finish_reason=length` 时记录截断且 Writer 继续；
- D + Agent 完整 Mock 流程；
- 最终校验失败仍保留完整临床进度；
- 14.2、11.8 与错误阈值 10 的区分；
- `µ/μ` 和空格差异的单位规范化；
- 只对短、明确、原文可证的 fact evidence 做安全修复；
- 推断阈值不会被静默改写为病例事实；
- 明确“无右上腹疼痛”不被误判；
- 未记载否定、事实反转、未检查写成阴性仍被拦截；
- 不同时间点的症状状态不被错误合并；
- Reviser 引入 `10μmol/L` 时仍会被发现并阻止进入最终成功结果。

本轮最终结果：120 项单元测试全部通过；Python 编译检查通过。

## 5. 真实复测

当前 Codex 进程没有继承 `DASHSCOPE_API_KEY`、`QWEN_BASE_URL` 和 `QWEN_MODEL`，因此本轮没有伪造新的真实运行结果，也没有静默回退 Mock。充值不等同于当前进程已获得密钥配置。

在已配置环境变量的 Mac 终端运行：

```bash
cd /Users/tintin7790/Code/Mewtype
python3.11 main.py \
  --input information/converted/02_input.json \
  --output information/results/02/d_agent_repaired.json \
  --log information/results/02/d_agent_repaired_log.json \
  --mode dev_api \
  --pipeline clinical_pipeline \
  --clinical-prompt-profile contrastive \
  --clinical-verification on \
  --clinical-revision on \
  --clinical-agents always \
  --graph-retrieval off \
  --clinical-response-format json_schema
```

该命令使用新文件名，不覆盖旧输出。复测时应重点核对：

1. `stage_calls` 中 Counter-Evidence Agent 是否完成、是否截断；
2. Writer 后 evidence 校验是否已出现 `10μmol/L`；
3. Reviser 后最终 evidence 校验是否新增该值；
4. “无右上腹疼痛”是否保持为来源支持的否定事实；
5. Agent、Verifier、Reviser 的调用次数、token、耗时和覆盖率；
6. 最终 output 的 evidence 编号、数值、单位、否定、时序及计划状态。

## 6. 仍需人工审核与已知限制

- 通过结构和事实一致性校验不等于诊疗建议医学正确；鉴别诊断和计划仍需医学人员审核。
- 自动 evidence 修复有意限制在短且明确的来源；长段落、跨来源推断和歧义指标不会被冒险改写。
- Compact Agent 在阿里云开发接口和赛事内网接口上的格式遵循、时延与 token 改善仍需真实复测。
- 既有失败日志无法回溯 `10μmol/L` 的首个生成阶段；只有新版本运行才能依据新增审计字段确认。
- 本地开发模型、量化、模板和服务参数与赛事模型可能不同，结果不能直接外推。
