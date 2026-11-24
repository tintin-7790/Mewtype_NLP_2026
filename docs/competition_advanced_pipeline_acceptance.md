# 正式入口与完整 D+Graph+Agent 统一验收

版本：`0.4.7-competition-profile-acceptance`
验收日期：2026-11-24
结论边界：以下完整链路使用本地 Mock Chat Completions 和确定性 Mock Embedding，只证明程序控制流、协议、预算及审计可运行，不代表官方 Qwen 或 BGE-M3 已通过测试。

## 1. 当前正式入口

`python main.py` 仍读取包根目录 `.env/input.json`，生成 `output.json/run_log.json`。当前打包常量为：

```python
ACTIVE_COMPETITION_PROFILE = "baseline_safe"
```

因此当前默认正式算法仍是一次 Qwen 调用的 Baseline。选择逻辑位于 `src/competition_profile.py`，无参数入口在 `src/app.py` 中将 Profile 确定性映射为已有 Pipeline 参数。`.env` 继续只要求：

```dotenv
MODEL_NAME=...
ENDPOINT=...
```

带参数的既有开发命令不经过正式 Profile，不受影响。不存在散落在 Planner、Writer 或模型客户端中的“强制 Baseline”逻辑；唯一默认选择是可审计的打包常量。

## 2. 内部 Profile

| Profile | Pipeline | 阶段 | 保留 Qwen 调用 | Embedding |
|---|---|---|---:|---|
| `baseline_safe` | baseline | Writer | 1 | 无 |
| `clinical_d` | clinical | Planner、Writer、Verifier、必要时 Reviser | 4 | 无 |
| `clinical_d_agent` | clinical | 上述阶段 + Counter-Evidence Agent | 5 | 无 |
| `clinical_d_graph_agent` | clinical | D + Agent + PPR | 5 | 官方 Dense 必需 |

Clinical Profile 默认使用 contrastive Planner、Verifier on、Reviser on；Agent 档为 `always`，Graph 档为 `ppr`。启动时重新计算实际最小阶段预算，若与 Profile 声明不一致，任务级失败。重试同样占用显式预算，不存在无限循环。

`clinical_d_graph_agent` 的正式 Embedding 边界只能构造 `OfficialEmbeddingClient`。协议未公布时该客户端联网前明确报告 pending；Pipeline 可以披露 `graph_retrieval_failed:<code>` 并降级为无 Graph 的 D+Agent，而不会把 Graph 标为已执行。Agent 自身失败继续沿用现有非阻断回到 Writer 的策略。Planner 只在剩余预算允许时使用既有限次回退。

Clinical Profile 额外启用最终安全门：若局部修订/确定性保守修复后仍存在 `unresolved_risk_count > 0`，该病例以 `UNRESOLVED_HIGH_RISK` 失败，不把未解决的重大患者事实错误包装成成功结果。MEDIUM/LOW 风险仍按既有保守待核策略处理。

## 3. 隔离完整链路验收

集成测试复制运行包到临时目录，使用正式数组格式的虚构 `DEMO001`，写入仅含 `MODEL_NAME/ENDPOINT` 的 `.env`，并实际启动：

```bash
python main.py
```

测试副本在 Python 启动时把打包 Profile 固定为 `clinical_d_graph_agent`，并用临时 `sitecustomize.py` 将尚未公布的官方 Embedding 边界替换为 `MockEmbeddingClient`。该文件只存在于临时测试目录，不在源代码、正式包白名单或运行配置中。Qwen 边界连接本地假 Chat Completions HTTP 服务；每个阶段返回与自身 Schema 匹配的独立响应。

验收结果：

| 检查项 | 结果与证据 |
|---|---|
| Planner | `stage_calls[0].stage=planner`，有效 Fact ID 规划 |
| Graph 构建 | 日志含 Problem–Fact/Source 图及边类型 |
| PPR | `graph_executed=true`、`graph_mode=ppr`、`ppr_iterations=20` |
| Graph → Agent | Agent HTTP 载荷候选含 `graph_strategy=ppr` |
| Agent | `completed=true`，完整覆盖 PR001，检索候选数大于 0 |
| Graph → Writer | Writer 的 `problem_evidence_context` 含 PPR 候选字段 |
| Verifier | 完整覆盖 C001–C005，并返回 HIGH 风险建议问题 |
| Reviser | `revision_attempted=true`、`revision_applied=true`，只替换目标 Plan Claim |
| Final Validation | `unresolved_risk_count=0`，格式/来源/数值/否定/状态校验通过 |
| 输出 | DEMO001 `status=success`，生成合法 `output.json` |

阶段调用顺序严格为：

```text
planner → counter_evidence → writer → verifier → reviser
```

共 5 次 Mock Qwen HTTP 调用。PPR 即使没有改变最终排序也会如实记录；本次断言的是传播确实执行、候选确实进入两个下游阶段，而不是只检查命令返回 success。

## 4. 降级与预算决策

正式提交前必须取得官方限制并选择 Profile：

1. 若只允许单次调用，保持 `baseline_safe`。
2. 若允许 4 次且无 Embedding，评估 `clinical_d`。
3. 若允许 5 次但无 BGE-M3，评估 `clinical_d_agent`。
4. 只有 5 次 Qwen、上下文/Token/超时均满足，且官方 Dense API 验收通过时，才选择 `clinical_d_graph_agent`。

Graph/Agent 是可披露降级模块；结构错误、非法来源以及未修复 HIGH 医学事实风险不是可忽略故障。任何降级均在 `algorithm_profile`、`clinical.algorithm_mode`、`clinical.graph_retrieval`、`clinical.adaptive_agent` 和阶段调用日志中记录实际行为。

## 5. 官方接口公布后的切换

1. 在 `OfficialEmbeddingClient` 边界按赛方文档实现请求/响应、1024 维、批量、超时、重试和预算；不把地址或认证塞入 Qwen `.env`，其固定配置形式需以官方材料为准。
2. 用单例验证普通 JSON、Qwen usage/finish_reason、BGE 数量/维度及错误处理。
3. 用同一固定病例集验证 5 阶段调用、PPR、Agent、Writer、Verifier、Reviser、耗时与总 Token。
4. 确认额度后，将 `src/competition_profile.py` 中唯一的 `ACTIVE_COMPETITION_PROFILE` 改为 `clinical_d_graph_agent`。
5. 重跑全量测试、真实接口兼容测试、独立解压复现和白名单审计，再冻结提交包。

当前没有切换：默认提交 Profile 仍为 `baseline_safe`。真实 Qwen+BGE-M3 仍等待 Endpoint/模型、Embedding 协议、认证方式、配额、上下文长度、结构化输出能力、超时与重试规则。

## 6. 本轮自动化结果

- 全量 unittest：150 项通过（原有 146 项全部保留，新增 Profile、完整链路、Embedding 故障降级和 HIGH 风险门测试）。
- Python 3.11 编译检查：通过。
- Baseline 无参数正式入口：通过。
- `clinical_d_graph_agent` 无参数隔离 Mock 入口：5 阶段、PPR、Agent、Verifier、Reviser 和最终输出全部通过。
- Embedding 故障降级：Graph 明确记录未执行及错误码，合法 D+Agent 输出保留；没有冒充 Graph 成功。
- 临时包白名单：62 个文件、单层根目录；无 tests、`sitecustomize.py`、真实 `.env`、病例、模型权重、缓存或历史输出。
- 未生成正式提交压缩包，未调用真实官方服务。
