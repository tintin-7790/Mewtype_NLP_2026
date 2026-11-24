# 统一两参数入口与完整 Pipeline 验收

版本：`0.4.8-unified-profile-entrypoint`
验收日期：2026-11-24
医疗用途声明：仅用于教学比赛和虚构/获授权数据测试，不用于真实诊疗。

## 1. 单命令选择规则

统一命令为：

```bash
python3.11 main.py --input input.json --output output.json
```

若没有显式传入 `--mode/--pipeline/--graph-retrieval/...` 等高级算法参数，入口自动应用运行 Profile：

| 条件 | 环境 | Profile | 模型边界 | Embedding |
|---|---|---|---|---|
| `MEWTYPE_DEV_MODE=1` | `dev_api` | `development_clinical_d_graph_agent` | 百炼兼容接口 | 明确标记的 Mock，仅开发 |
| 未设置开发开关 | `competition` | 打包常量，当前 `baseline_safe` | `.env` 的赛方完整 Endpoint | Baseline 不使用 |
| 显式高级 CLI | 由 CLI 决定 | `explicit_cli` | 保持既有语义 | 由显式配置决定 |

开发模式不会因 `DASHSCOPE_API_KEY`、`QWEN_BASE_URL` 或 `QWEN_MODEL` 恰好存在而自动开启，只有值严格为 `1` 的 `MEWTYPE_DEV_MODE` 才触发。competition 模式不读取百炼密钥、不发送外部 Authorization、不使用 Mock Embedding。正式 `.env` 仍只需 `MODEL_NAME/ENDPOINT`。

开发 Profile 固定配置：Clinical Pipeline、contrastive、Verifier on、Reviser on、Agent always、PPR、阶段化 `json_schema`，每例预留 5 次 Qwen 调用。高级 CLI 参数全部保留。

## 2. 输入自动识别

默认 `--input-format auto` 仍使用唯一的严格检测器：

- JSON 数组进入正式 Sources 解析器，原样保留 `case_id`、`record_cutoff`、顺序、`source_id`、`field_path` 和内容。
- 顶层唯一 `入院记录` 的对象进入练习适配器，在内存中展平，不生成中间输入文件，不修改原文。
- 未知、混合或损坏 JSON 返回任务级明确错误，不猜测来源或事实。
- 练习文件带可靠前导序号时保留 `PRACTICE07` 一类编号；改名为 `input.json` 时使用脱敏内容指纹。指纹计算前替换姓名、证件号、住院号、电话、地址、工作单位等直接身份字段，得到 `PRACTICE_AUTO_<12位SHA256>`。测试确认改变姓名不会改变 case_id。
- `record_cutoff` 只读取合法的 `入院记录.一般项目.记录时间` 或显式覆盖，不使用当前系统时间。

## 3. 第 07 份文件检查结果

原文件：`information/临床智录_练习病历数据集JSON/07_心血管内科_多次PCI术后再发胸痛.json`

该文件当前不是合法 JSON。Python 3.11 报告：

```text
Extra data: line 67 column 1 (char 3660)
```

文件第 66 行已经闭合顶层对象，第 67 行又出现 `]`。测试把原始字节复制为隔离目录的 `input.json`，直接运行统一两参数入口；结果为 `INPUT_INVALID_JSON`、退出码 2、没有生成 `output.json`、没有发起模型调用。程序未删除尾部字符、未静默截取第一个 JSON 值、未修改原文件。

因此本轮不能诚实声称第 07 份原文件已经完成 D+Graph+Agent。多次 PCI 日期顺序、胸痛发生/变化/缓解、抗血小板药漏服、待排诊断、患者自述肺部阴影及检查状态等病例级不变量，必须在资料提供方交付合法 JSON 后再通过同一入口验证。适配器只负责原样保留字段路径和值，不在代码中硬编码这些病例答案。

## 4. 离线完整链路验收

### 有效原始练习 JSON

隔离目录内使用文件名 `input.json` 的合法嵌套练习记录，设置开发开关后执行统一命令。结果：

- 自动识别为 `practice`。
- 自动生成脱敏稳定 case_id。
- Unicode 日期连字符被审计性规范化，原始值保留在适配日志。
- Mock Dense、PPR 和 Counter-Evidence Agent 实际执行。
- Planner、Agent、Writer、Verifier 均使用各自 `json_schema`。
- 输出通过最终校验。

### DEMO001 正式 Sources

同一两参数入口使用 DEMO001 的 S01–S04 数组，保留四个来源编号与顺序。隔离 Mock Chat Completions 为各阶段分别返回合法 Schema：

```text
planner → counter_evidence → writer → verifier → reviser
```

验收证据：

- `graph_executed=true`、`graph_mode=ppr`、`ppr_iterations=20`。
- Agent 请求中的候选包含 `graph_strategy=ppr`。
- Writer 的 `problem_evidence_context` 包含同一 PPR 候选。
- Verifier 完整覆盖 Claims，并以 HIGH 风险触发一次局部 Reviser。
- `revision_attempted=true`、`revision_applied=true`、`unresolved_risk_count=0`。
- Final Validation 通过并生成合法 `output.json`。

以上均为本地 Mock 协议/控制流验收，不代表 Mock Embedding 具备 BGE-M3 语义质量，也不代表官方模型测试成功。

## 5. competition 与 dev_api 隔离

competition 两参数测试在进程环境中故意放入百炼变量，但未设置 `MEWTYPE_DEV_MODE`。程序仍选择 `baseline_safe`、读取正式 `.env`、请求赛方 Mock Endpoint，HTTP 请求没有 Authorization，且只发送最小 Chat Completions 字段。这证明环境中“碰巧存在百炼密钥”不会改变正式模式。

开发 Profile 的 Mock Embedding 由内部 Profile 明确构造并记录为 `mock_development / development_mock_only`。正式 Profile 只有在打包常量切换到 `clinical_d_graph_agent` 时才构造 `OfficialEmbeddingClient`；其协议仍待赛方公布，绝不自动换成开发 Mock。

## 6. Prompt A/B 与输出保护

当前 A 版保持现有 contrastive Prompt 文件不变。每次日志新增：

- `prompt_version`；
- 实际使用的 Prompt 相对路径和 SHA-256；
- 模型标识与非敏感生成参数；
- `algorithm_profile`、输入病例标识；
- 每阶段 Token、耗时、解析与 Schema 状态；
- Graph/PPR、Agent、Verifier/Reviser 和最终验证审计。

开发 Profile 若发现目标输出或同目录 `run_log.json` 已存在，会先复制为：

```text
<原文件名>.backup.<UTC时间>.<run_id前8位>
```

再原子写入新结果。测试确认两个历史文件均被保存。后续 v2 Prompt 必须使用同一输入、Profile、模型参数和服务条件，并写到新的结果路径或依赖上述备份，不覆盖 A 版。

## 7. 真实百炼测试命令

本轮没有主动发起任何付费 Qwen 请求。用户已在自己的 Mac 终端配置百炼变量后，只需执行一次：

```bash
cd /Users/tintin7790/Code/Mewtype
export MEWTYPE_DEV_MODE=1
python3.11 main.py --input input.json --output output.json
```

该命令会自动运行真实百炼 Qwen + 开发 Mock Dense/PPR + Agent/Verifier/Reviser，并把运行环境标记为本地开发验证。若输入使用当前损坏的第 07 文件，会在模型调用前明确失败；应先向资料提供方取得合法 JSON，不能手工静默修补后冒充原始输入。

## 8. 自动化结果与待验证项

- 全量 unittest：153 项通过。
- Python 3.11 编译检查：通过。
- competition 两参数入口、有效原始 Practice、DEMO001 Sources、完整 Graph+Agent、环境隔离、备份机制：通过。
- 原始第 07 文件：正确拒绝，错误位置已固定回归。
- 未覆盖历史 PRACTICE01/02/03 输出，未生成正式提交 ZIP。

仍待赛方公布并验证：正式 Qwen 模型/Endpoint、BGE-M3 请求响应与认证、1024 维和批量限制、调用与 Token 预算、上下文长度、并发、超时重试、`json_schema` 支持。正式默认仍为 `baseline_safe`；只有完整官方验收后才切换打包常量到 `clinical_d_graph_agent`。
