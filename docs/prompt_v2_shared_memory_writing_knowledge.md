# Prompt v2、临床写作知识与 Shared Clinical Memory 接入报告

## 1. 实施边界

本报告记录三个彼此独立模块最初接入时的离线验收。自 `0.4.9` 当前开发配置起，Prompt 已正式选定为 `contrastive_v2`，完整本地开发 Profile 默认同时启用写作知识和 Shared Clinical Memory；普通高级 CLI 与正式 competition Profile 仍默认关闭这些附加模块。

最初接入的三个模块为：

1. 五阶段 `contrastive_v2` Prompt；
2. Writer 临床写作规则与原创合成示例风格提要；
3. 单病例 Shared Clinical Memory。

没有修改 `ACTIVE_COMPETITION_PROFILE="baseline_safe"`、正式输入输出 Schema、五个阶段 JSON Schema、当前稳定 `contrastive` Prompt 或历史实验结果。没有调用 Qwen、Embedding 或其他付费/外部服务。

## 2. ZIP 审查与完整性

两个 ZIP 均先解压到临时目录并阅读实施指南。临床写作资源包的 `SHA256_MANIFEST.json` 与六份资源文件全部匹配。

Prompt v2 ZIP 自带的 `prompt_sha256.json` 与同一 ZIP 中实际五份 Prompt **全部不匹配**。这不是代码接入造成的差异。例如其声明 Planner 为 `418943...`，实际 ZIP 成员为 `ff7270...`。因此项目没有把原清单冒充验证成功，而是在 `prompts/contrastive_v2/prompt_sha256.json` 中登记集成时对实际成员计算的哈希，并保留完整性说明。后续如提供方发布修正版，需先逐文件 diff 后再更新。

## 3. 第一步：五阶段 Prompt v2

独立目录：`prompts/contrastive_v2/`。`src/prompt_profiles.py` 统一解析以下阶段：

| 阶段 | Prompt | 服务端 Schema |
|---|---|---|
| Planner | `planner_system.txt` | `PLANNER_SCHEMA` |
| Counter Agent | `counter_evidence_system.txt` | `COUNTER_EVIDENCE_SCHEMA` |
| Writer | `writer_system.txt` | `MEDICAL_SCHEMA` |
| Verifier | `claim_verify_system.txt` | `VERIFIER_SCHEMA` |
| Reviser | `revise_system.txt` | `REVISER_SCHEMA` |

启用方式：

```bash
--clinical-prompt-profile contrastive_v2
```

五阶段完整 Mock 路由测试已经实际运行，调用顺序为 Planner → Counter Agent → Writer → Verifier → Reviser，全部继续通过各自原有结构校验。v1 的 `contrastive`、`contrastive_fewshot` 与 `legacy` 路径保持原样。

离线字符比较（相同加载方式，不是模型 Token）：

| 阶段 | v1 system chars | v2 system chars | 差值 |
|---|---:|---:|---:|
| Planner | 1853 | 1450 | -403 |
| Counter Agent | 1063 | 1072 | +9 |
| Writer | 1951 | 1517 | -434 |
| Verifier | 1823 | 1505 | -318 |
| Reviser | 1136 | 1024 | -112 |

真实 Token 不能由字符数代替。真实运行时每阶段 `stage_calls.token_usage`、`elapsed_ms`、模型与 response format 会继续记录。

## 4. 第二步：Writer 写作知识

六份资源位于 `resources/clinical_writing/`，哈希、来源和用途已登记到 `resource_manifest.json`。运行时纯 Python、无网络、无新增依赖。

模式：

- `off`：完全不增加 Writer 输入；
- `rules`：检索确定性的 style cards；
- `rules_examples`：规则卡加最多一个合成示例的 `style_takeaways`。

示例选择不会返回合成病例 `case` 或 `medical_record`，不会将示例数值、药物、日期、诊断或 source_id 作为当前病例内容。结果放在独立的 `writing_style_context`，不进入 `sources`、`facts`、`evidence` 或 Verifier。

小型 PCI/胸痛夹具的离线 Writer payload：

| 模式 | payload chars | guidance chars | 准备耗时 ms（单次本机观察） |
|---|---:|---:|---:|
| off | 1141 | 0 | 0.140 |
| rules | 2556 | 1372 | 0.179 |
| rules_examples | 2665 | 1480 | 0.277 |

该耗时只是本机单次离线观察，不是性能承诺。正式比较应使用同一病例、模型与参数，并查看真实 Prompt Token。

## 5. 第三步：Shared Clinical Memory

`src/shared_clinical_memory.py` 创建每例独立的进程内对象：

- evidence index：只保留当前病例 Fact/Source ID 及结构属性；
- Planner 写入 problem decision cards；
- Graph/PPR 将候选来源映射回 Fact ID；
- Counter Agent 写入增量 issue ID；
- Writer 读取紧凑 decision cards，并被明确要求回到 Sources 核实。

Verifier 和 Reviser 函数不接收 Shared Memory 参数，仍以当前病例原始 Sources 为唯一事实边界。Memory 不写入 output.json、不落盘、不跨 case_id、不增加模型调用。

小型夹具中，Writer payload 从 1141 增加到 1587 字符；Memory 自身一次计划写入和一次渲染累计约 0.016 ms。运行日志记录 `fact_index_count`、`decision_card_count`、更新/渲染次数和耗时，不记录完整病例正文。

## 6. 测试

新增 10 项模块测试：

- 五份 v2 独立加载与实际哈希；
- 三个研究开关默认关闭且正式 Profile 未改变；
- v1 路径不变；
- v2 与五阶段既有 Schema 契约一致；
- 写作规则确定性、长度边界及非证据标识；
- 合成示例只返回风格提要；
- Writer 三种开关组合隔离；
- Shared Memory 病例隔离；
- Planner/Graph/Agent 卡片更新；
- 三阶段与完整五阶段 Mock 路由、Token/耗时日志。

全量测试为 `176/176` 通过。Python 3.11 编译、资源 JSON 解析与 `git diff --check` 通过。测试全部离线。

## 7. 历史 A/B 命令（不再是当前运行要求）

以下命令仅保留作历史研究记录，当前不再安排 Prompt v1/v2 对照。日常本地运行应设置 `MEWTYPE_DEV_MODE=1` 后使用统一两参数命令。本轮没有调用付费 API。

### 7.1 Prompt v1 对 v2（其他两模块关闭）

```bash
python3.11 main.py --input test_input.json --output ab_prompt_v1.json --log ab_prompt_v1_log.json \
  --mode dev_api --pipeline clinical_pipeline --clinical-prompt-profile contrastive \
  --clinical-writing-guidance off --shared-clinical-memory off \
  --clinical-agents off --clinical-verification on --clinical-revision on \
  --clinical-response-format json_schema

python3.11 main.py --input test_input.json --output ab_prompt_v2.json --log ab_prompt_v2_log.json \
  --mode dev_api --pipeline clinical_pipeline --clinical-prompt-profile contrastive_v2 \
  --clinical-writing-guidance off --shared-clinical-memory off \
  --clinical-agents off --clinical-verification on --clinical-revision on \
  --clinical-response-format json_schema
```

### 7.2 写作知识消融（Prompt 固定 v2，Memory 关闭）

```bash
# C0: off
python3.11 main.py --input test_input.json --output ab_write_off.json --log ab_write_off_log.json \
  --mode dev_api --pipeline clinical_pipeline --clinical-prompt-profile contrastive_v2 \
  --clinical-writing-guidance off --shared-clinical-memory off \
  --clinical-agents off --clinical-verification on --clinical-revision on \
  --clinical-response-format json_schema

# C1: rules；另可把 rules 改成 rules_examples 做第三臂
python3.11 main.py --input test_input.json --output ab_write_rules.json --log ab_write_rules_log.json \
  --mode dev_api --pipeline clinical_pipeline --clinical-prompt-profile contrastive_v2 \
  --clinical-writing-guidance rules --shared-clinical-memory off \
  --clinical-agents off --clinical-verification on --clinical-revision on \
  --clinical-response-format json_schema
```

### 7.3 Shared Memory 消融（写作知识关闭）

复制开发 Mock 检索配置后，保持两轮相同：

```bash
cp config/retrieval.mock.example.json config/retrieval.mock.json

# D0: Memory off
python3.11 main.py --input test_input.json --output ab_memory_off.json --log ab_memory_off_log.json \
  --mode dev_api --pipeline clinical_pipeline --clinical-prompt-profile contrastive_v2 \
  --clinical-writing-guidance off --shared-clinical-memory off \
  --retrieval-config config/retrieval.mock.json --graph-retrieval ppr --clinical-agents always \
  --clinical-verification on --clinical-revision on --clinical-response-format json_schema

# D1: Memory on
python3.11 main.py --input test_input.json --output ab_memory_on.json --log ab_memory_on_log.json \
  --mode dev_api --pipeline clinical_pipeline --clinical-prompt-profile contrastive_v2 \
  --clinical-writing-guidance off --shared-clinical-memory on \
  --retrieval-config config/retrieval.mock.json --graph-retrieval ppr --clinical-agents always \
  --clinical-verification on --clinical-revision on --clinical-response-format json_schema
```

Mock Embedding 仅用于固定 Graph/PPR 控制流，不能评价 BGE-M3 语义质量。评价需分别比较格式成功率、患者事实错误、关键时序/否定/状态、证据支持比例、重要遗漏、Prompt/Completion Token、各阶段耗时和调用数；不得把多项开关同时变化后归因给某一个模块。

## 8. 待真实验证

- v2 Prompt 尚未在 Qwen 上做 A/B，不能声称质量提升。
- 写作规则增加 Writer 上下文，是否带来结构收益或事实串扰需要人工医学评审。
- Shared Memory 是否降低重复 Token、改善 Graph/Agent 协作尚待真实固定病例实验。
- Prompt ZIP 原哈希清单不一致需要资源提供方确认。
- 写作资源包声明为原创合成且不含真实患者信息；正式比赛打包前仍应由团队最终确认再分发许可与赛方资源申报要求。
