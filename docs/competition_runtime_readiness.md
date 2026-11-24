# 企业赛正式运行就绪报告

版本：`0.4.6-competition-runtime-readiness`
验收日期：2026-11-24
范围：Qwen Chat Completions 协议与运行工程验收；不代表真实赛方模型效果验收。

## 1. 正式入口

提交包根目录准备 `.env` 与 `input.json` 后执行：

```bash
python main.py
```

无参数档固定为 `competition / baseline`，从程序包根目录读取：

```dotenv
MODEL_NAME=赛方模型名称
ENDPOINT=赛方提供的完整 Chat Completions 地址
```

输出为包根目录 `output.json`，运行日志为 `run_log.json`。输入采用正式比赛 JSON 数组并严格经过原有协议解析；输出逐例保留数量、顺序及 `case_id`，通过同目录临时文件原子替换。带参数的开发、Mock、本地及多阶段实验入口保持不变。

`ENDPOINT` 被原样传给 HTTP 客户端，不自动添加 `/v1` 或 `/chat/completions`。正式客户端请求头只有 `Content-Type: application/json`，不读取百炼密钥，不发送 Authorization。默认请求体仅发送 `model`、`messages`、`stream=false`；在赛方确认前不默认发送 `response_format`、`seed`、`temperature`、`top_p` 或 `max_tokens`。

## 2. 已验证行为

- Python 3.11 可运行，正式依赖仍为标准库。
- 本地假 HTTP 服务实际执行 `python main.py`，无需参数或交互即可生成合法输出与日志。
- 正确解析 `choices[0].message.content`、`usage` 与 `finish_reason`。
- 日志记录阶段、模型、响应格式、调用成功/解析状态、Token、耗时、错误和调用数，不记录密钥、病例全文或完整模型响应。
- 完整 URL 路径未被拼接，Mock 服务确认未收到 Authorization。
- 正常响应、模型 JSON 截断、HTTP 503、客户端超时均有明确结果；没有静默回退 Mock。
- 两病例测试保持输入顺序和 `case_id`，每例独立失败不会终止后续病例。
- 输出采用原子写入；空数组、协议错误及既有输入适配行为由全量回归覆盖。
- `OfficialEmbeddingClient` 在协议未公布前会在联网前明确失败；competition 模式拒绝本地 BGE/Mock，也拒绝自行杜撰的 official HTTP 字段。
- 打包白名单不包含 `.env`、`information/`、模型权重、缓存、虚拟环境、历史输出或日志，只包含 `.env.example`。

以上 HTTP 验收使用确定性本地服务，不是官方 Qwen 实测，不能据此声称 `json_schema`、生成质量、速度或配额兼容。

## 3. 调用预算与降级策略

已知 PRACTICE03 的 D+Graph+Agent 开发实验使用 5 次 Qwen 调用、74,772 Token。这不是官方预算，也不能作为正式默认值。

| 档位 | 主要 Qwen 阶段 | 最小预算 | 正式建议 |
|---|---|---:|---|
| Baseline | Writer | 1 | 未知配额下的无参数安全默认 |
| Clinical 最小链路 | Planner、Writer | 2 | 先确认两次调用及上下文上限 |
| D | Planner、Writer、Verifier，必要时 Reviser | 4（预留修订） | 只有赛方预算允许时启用 |
| D+Agent | Planner、Agent、Writer、Verifier，必要时 Reviser | 5 | Agent 为可选研究阶段，预算不足时启动前关闭 |

Graph/Dense 编码不增加 Qwen 阶段，但依赖尚未公布的 Embedding 服务及其独立预算。程序启动前按所选链路验证最小 Qwen 调用预算；每个 HTTP 重试也占用预算，循环均有上限。Planner 回退只在剩余预算足够时发生；Agent 失败会留痕并安全回到 Writer；Graph 不可用会明确记录 fallback，不能冒充执行成功。Verifier 的 LOW 不触发修订，MEDIUM 进入待核提示，HIGH 才触发至多一次 Reviser 或确定性保守回退。最终 Python 结构、来源、数值、否定、时序及状态错误仍会拦截。

正式配额公布后，应在运行前选择满足调用次数、上下文长度与超时限制的档位，而不是运行中擅自超限。无参数档当前有意固定一调用 Baseline；多阶段仍通过已有显式 CLI/JSON 配置做官方兼容验证。

## 4. 等待赛方公布

以下信息未在代码中猜测：

1. 正式 `MODEL_NAME` 与完整 `ENDPOINT`。
2. Chat Completions 的请求字段白名单、是否接受 `stream=false`、响应错误结构。
3. `text/json_object/json_schema` 支持范围及 Schema 限制。
4. 单例/全任务调用次数、Token、并发、超时、重试与上下文上限。
5. 是否返回标准 `usage` 和 `finish_reason`；当前缺失时可运行但统计为零/空。
6. BGE-M3 HTTP 地址、请求/响应字段、批量大小、1024 维约束、认证、超时、重试和配额。
7. 正式材料白名单、团队目录名、日志文件名及是否允许附带研究模块。

赛方公布后只需补充 `.env` 并先做单例协议探测；若请求字段或响应结构与 Chat Completions 标准存在明确差异，只修改模型边界适配器。Embedding 协议只在 `src/embedding_client.py` 的 `OfficialEmbeddingClient` 及其配置解析处实现，不改变病例、检索、核验与输出协议。

## 5. 正式开放后的验收顺序

1. 用非敏感最小病例验证完整 Endpoint、模型名、无认证请求和普通文本 JSON。
2. 核对 `choices/message/content`、错误响应、usage、finish_reason、超时和配额。
3. 只有赛方确认支持后，分别测试 `json_object` 与阶段化 `json_schema`，不配置静默回退。
4. 运行固定病例集，检查格式、来源、数值、否定、时序、计划状态、Token 和耗时。
5. 根据官方预算决定 Baseline、D 或 D+Agent；Embedding 未验收时保持 Graph/Dense 关闭。
6. 独立环境解压白名单包，放置正式 `.env/input.json`，再次执行 `python main.py` 并核对 `output.json/run_log.json`。

## 6. 剩余风险

- 官方服务可能要求或拒绝某些标准字段；当前只发送最小请求体，但 `stream=false` 仍需实测。
- `text` 模式依赖模型遵循 JSON 提示，可能产生截断或格式错误；在确认结构化输出能力前不能消除该风险。
- 无参数档采用 Baseline，不包含 D/Agent 的额外质检收益；这是未知预算下的保守工程选择，不代表最佳算法配置。
- 远程模型、模板、量化和生成实现不同于百炼 qwen-plus，开发效果不能外推。
- Official Embedding 尚不可调用；Graph 正式链路在协议公布和真实验收前必须保持关闭。
- 当前输出协议沿用旧详细规范；赛方若更新 Schema、文件命名或日志要求，需要调整协议边界和打包白名单。

## 7. 本轮验收记录

- `python3.11 -m unittest discover -s tests -v`：146 项通过。
- `python3.11 -m compileall -q main.py src tools tests`：通过。
- 无参数子进程测试：正常、截断 JSON、HTTP 503、超时和双病例顺序场景通过。
- 临时打包审计：60 个文件，只有一层 `MewtypeAudit_0.4.6/` 根目录；必需文件无缺失，敏感/本地资源命中数为 0。
- 未调用真实赛方 Qwen 或 BGE-M3；未生成正式提交压缩包，未修改历史实验输出。
