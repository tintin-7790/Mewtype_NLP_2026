# 北京语言大学Mewtype｜循证首次病程记录生成系统

**参赛队伍：** 北京语言大学Mewtype

**交付算法版本：** v1（程序版本 `0.4.9-prompt-v2-research-modules`）

**开发与验收环境：** Python 3.11

> 本程序仅用于教学比赛和技术评审，不能替代执业医师的诊断、治疗决策或病历签署。生成结果必须由具备相应资质的人员审核。

## 1. 项目概述

Mewtype 根据资料截止点前的入院资料，生成病例特点、拟诊讨论、诊疗计划、关键来源依据和待核事项。系统强调：

- 未提供不等于正常、否认或未执行；
- 初步诊断、疑似和待排不得升级为确诊；
- 已实施、进行中、原文计划和模型新增建议必须区分；
- 关键事实与推断分开，来源编号只能引用当前病例；
- 无法安全修订的高风险问题必须保留为未解决风险，不能用通用模板掩盖。

代码同时保留单次生成 Baseline 和分阶段 Clinical Pipeline。实际执行路径由运行 Profile 决定，不能因为代码包含某个研究模块就声称正式评测已经启用该模块。

## 2. 正式启动

在提交包的一层团队目录中安装依赖。按最新提交通知使用
`requirements.txt`；`requires.txt` 作为旧通知兼容副本保留，两者内容一致：

```bash
python -m pip install -r requirements.txt
```

随后按主办方规范无参数启动：

```bash
python main.py
```

程序固定读取 `main.py` 所在目录的 `input.json` 和 `.env`，并在同一目录
生成 `output.json` 与 `run_log.json`。无需交互、额外脚本或本地服务。

以下两参数形式继续保留，用于开发和兼容测试：

```bash
python main.py --input input.json --output output.json
```

相对输入、输出和日志路径按命令执行时的当前工作目录解析；Prompt 和随包资源按程序目录定位。程序无需交互，默认在输出文件同目录生成 `run_log.json`，并通过同目录临时文件原子替换结果文件。

正式运行从程序根目录 `.env` 读取：

```dotenv
MODEL_NAME=赛方提供的模型名称
ENDPOINT=赛方提供的完整Chat-Completions地址
```

`.env` 正式运行只需要上述两项。`ENDPOINT` 按原值使用，不自行拼接路径；`MODEL_NAME` 原样写入请求体的 `model` 字段。正式客户端不读取 `DASHSCOPE_API_KEY`，不发送 `Authorization`，也不会在失败时回退到百炼、本地模型或 Mock。即使宿主进程残留 `MEWTYPE_DEV_MODE` 或百炼变量，无参数正式入口仍强制使用 competition Profile。

## 3. 正式 Profile 与本地完整 Profile

Profile 固定在 `src/competition_profile.py`，不要求赛方在 `.env` 中增加算法字段。

### 正式 competition Profile

当前 `ACTIVE_COMPETITION_PROFILE="competition_full_v2"`：

- Pipeline：`clinical_pipeline`
- Prompt：`contrastive_v2`
- 检索：纯标准库 Lexical 种子 + 病例内临床图 PPR
- Counter-Evidence Agent：每例执行
- Writing Guidance：`rules_examples`
- Shared Clinical Memory：启用，仅限当前病例内存
- Claim Verifier 与 Local Reviser：启用；Reviser 仅在 HIGH 风险时最多调用一次
- Final Validation：`lite`，明确患者事实错误和结构错误仍为 HARD
- Qwen 调用预算：每例最多 5 次；无修订时通常 4 次
- Response format：`text`
- 模型来源：仅赛方提供的 Qwen Chat Completions Endpoint

正式 Profile 不发送 `response_format`，避免假定赛方支持百炼的 `json_schema`。Lexical Graph 不需要 BGE、本地模型或第三方依赖。`baseline_safe` 仍作为可选择的独立构建 Profile 保留；实际运行 Profile 会明确记录在 `run_log.json`，不会静默伪装为其他算法。

正式 Chat Completions 生成策略集中在 `src/config.py`：HTTP 等待上限默认 900 秒；Planner、Counter-Evidence、Verifier、Reviser 为 8192 `max_tokens`，Writer 为 12288。Planner、Counter-Evidence 和 Writer 默认通过 `chat_template_kwargs.enable_thinking=true` 开启思考，Verifier/Reviser 默认关闭以优先保证紧凑 JSON 完整；只有一次预算的 baseline 路径不发送可选 Thinking 参数。`temperature=0`、`top_p=1`、`seed=20261125`、`stream=false` 均随请求发送。`max_tokens` 是生成上限而非目标篇幅。若服务明确以 HTTP 400/422 拒绝 Thinking 参数，客户端只允许一次计入五次预算且有剩余预算时才执行的去参数兼容重试，并对后续阶段缓存降级；不会无限重试。非流式为正式默认，显式配置 `stream=true` 时客户端可聚合 OpenAI 兼容 SSE，始终只采用 `content`，不把 `reasoning_content` 写入病历 JSON。全部有效参数与兼容降级记录在 `run_log.json`；`.env` 仍只含 `MODEL_NAME` 和 `ENDPOINT`。

### 本地开发 Profile

仅当开发者明确设置 `MEWTYPE_DEV_MODE=1` 时，同一条两参数命令才选择 `development_clinical_d_graph_agent`：

- Pipeline：`clinical_pipeline`
- Prompt：`contrastive_v2`
- Clinical Planner、Counter-Evidence Agent：启用
- Graph Retrieval：PPR
- Embedding：确定性 Mock，仅验证控制流，不代表 BGE-M3
- Writer writing guidance：`rules_examples`
- Shared Clinical Memory：启用，仅限当前病例内存
- Verifier/Reviser：启用，最多一次局部修订
- 各模型阶段 Response format：`json_schema`
- 模型来源：现有百炼 `dev_api` 适配器

本地运行需要开发者自行在终端配置 `DASHSCOPE_API_KEY`、`QWEN_BASE_URL` 和 `QWEN_MODEL`。程序不会记录密钥，也不会仅因发现这些变量而自动进入开发模式。开发 Profile 会先备份同名历史输出和日志。

```bash
export MEWTYPE_DEV_MODE=1
export DASHSCOPE_API_KEY='...'
export QWEN_BASE_URL='https://.../compatible-mode/v1'
export QWEN_MODEL='qwen-plus'
python3.11 main.py --input input.json --output output.json
```

开发 Mock、百炼和本地模型均不得用于正式 competition 环境。高级 CLI 参数继续保留用于受控实验，显式算法参数优先于 Profile。

## 4. 输入与输出协议

默认 `--input-format auto` 严格识别：

1. 正式 Sources JSON 数组：每例包含 `case_id`、`record_cutoff`、`sources`；来源包含 `source_id`、`content`，可包含 `field_path` 等元数据。程序保留病例顺序、病例 ID、截止点、来源编号和原文。
2. 中文嵌套练习病历：顶层包含 `入院记录`。程序复用统一适配器在内存中展平，确定性生成来源编号并保留字段路径；不修改原始文件，不写中间病例文件。

未知、混合、损坏或仅部分符合正式字段的输入会明确失败，不会通过宽松猜测生成患者事实。练习数据缺少可靠记录时间时必须显式提供截止点，不能使用系统当前时间猜测。

输出为 JSON 数组，顺序和 `case_id` 与输入对应。每例包含：

- `case_id`
- `status`：`success` 或 `failed`
- `medical_record`：成功时包含 `case_features`、`diagnostic_discussion`、`plan`、`evidence`、`review_items`
- `error`：失败时包含错误码和消息

单例失败不会中断后续合法病例。结果文件不会混入 Markdown、解释文字或 HTTP 原始响应。

## 5. 算法与质量控制

### Baseline

`baseline_safe` 作为可独立选择的保守回退链路：原始 Sources → 一次 Qwen 生成 → Schema、来源、数值/单位、否定、计划状态和文本质量检查 → 输出。

### 正式 Clinical Pipeline

当前正式 `competition_full_v2` 链路包括：事实编号 → Clinical Planner → 词法检索和病例内临床图 PPR → Counter-Evidence Agent → Clinical Writer → Claim Verifier → 必要时最多一次 Local Reviser → 确定性 Lite Final Validation → Output Guarantee。

Output Guarantee 是唯一的降级输出选择层：Verifier 和 Reviser 只产生问题、修订与审计信息，不会因解析失败、超时或单个 HIGH 标记直接清空整份病历。该层依次选择完整增强输出、安全局部修订、保守净化的 Writer 输出，或仅基于当前病例 Sources 的确定性保守草稿。已定位的危险建议或事实矛盾仍必须被精确删除、保守改写或转入 Sources Fallback，不会为了返回 `success` 而原样放行。日志记录实际恢复级别、触发原因、移除或净化的 Claim ID 及人工复核需求。

Dense/词法相似度和 PPR 只用于证据定位，不代表医学事实成立，也不会自动覆盖模型声明的 `source_ids`。Shared Clinical Memory 是当前病例进程内决策卡，不跨病例持久化；写作规则和合成示例只提供结构、语气参考，不是患者事实。Verifier、Reviser 和 Final Validation 始终返回当前病例原始 Sources 核验。

局部修订必须精确命中完整句级 Claim，并通过原文匹配、范围和修改后结构检查。正文出现括号残缺、通用审稿模板或不可操作的 `review_items` 会被最终质量门拒绝。

## 6. Embedding 与受控降级

正式 Graph Profile 设计了三种检索策略：

- `dense_graph`：正式 BGE-M3 Dense Embedding + PPR；赛方协议公布并联调后才能启用。
- `lexical_graph`：纯 Python 标准库、病例内 BM25 风格词法种子 + PPR。
- `no_graph`：关闭图检索，保留 Planner、Agent、Writer、Verifier/Reviser。

正式环境绝不把开发 Mock Embedding 当作真实检索。当前正式 `competition_full_v2` 直接采用已离线验收的 `lexical_graph`，不会调用未公布的 BGE 接口。当前 `OfficialEmbeddingClient` 仍是明确失败的协议边界；未来若改用 Dense，必须先验证官方 Embedding。词法 Graph 运行失败时按已测试策略降级为 `no_graph` 并记录原因，不能伪装为已执行。

## 7. 资源与依赖

正式运行路径只使用 Python 3.11 标准库，包括 `urllib` 实现的 HTTP 客户端，因此 `requirements.txt` 和兼容副本 `requires.txt` 均没有第三方 PyPI 包。注释中的 Python 版本说明不是 pip 依赖条目。可选的 `mlx_lm` 和 `FlagEmbedding` 仅在开发者显式选择相应本地适配器时延迟导入，不属于正式提交依赖，也不随包安装模型权重。

随包 Prompt、归档兼容 Prompt、Few-shot 合成示例和临床写作资源均登记在 `resource_manifest.json`，并记录 SHA-256。临床写作资源不包含真实患者资料，不是外部医学知识库，也不能作为当前病例的证据来源。

## 8. 日志与可复现性

`run_log.json` 记录 run_id、程序与 Prompt 版本、实际 Prompt 哈希、运行环境、算法 Profile、模型标识、非敏感参数、每阶段调用次数、Token、耗时、Graph/Agent 审计、核验/修订状态和错误。普通日志不复制完整病例、完整模型响应或密钥。

确定性适配、规则检查、词法检索和排序可复现；远端 Qwen 受服务版本和推理实现影响，不承诺逐字一致。只有明确确认属于虚构病例并显式开启调试时，才能把原始响应写入独立调试文件。

## 9. 测试与打包

源代码工作区包含测试时，可执行：

```bash
python3.11 -m unittest discover -s tests -p 'test_*.py'
python3.11 -m compileall -q main.py src tools tests
```

正式算法 ZIP 不包含 `tests/`，因此上述命令不是赛方运行前置步骤。打包命令为：

```bash
python3.11 tools/build_package.py --team 北京语言大学Mewtype --version v1
```

打包脚本生成一层 `北京语言大学Mewtype_v1/` 目录，排除 `information/` 原始病例、历史输出、日志、密钥、缓存、虚拟环境、本地权重和无关实验文件。

## 10. 已知限制与提交前待确认

1. 赛方 Qwen 的真实 Endpoint、模型名称、超时、每例至少 4 次且最多 5 次调用的允许性、上下文长度和 Token 配额仍需正式环境联调。
2. 赛方 BGE-M3 HTTP 请求/响应、认证、维度和配额尚未公布；当前正式完整 Profile 使用 Lexical Graph，不依赖 Embedding。
3. 正式输入 Schema、Python/Linux 资源限制及材料命名以赛方最终通知为准。
4. 若赛方调用预算不足以运行 `competition_full_v2`，应在打包前明确切换为保留的 `baseline_safe` 或其他已验收 Profile，不能在运行中隐藏实际算法。
5. 本系统输出仅为教学比赛草稿，医学正确性仍需专业人员审核。
