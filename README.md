# 循证智录：首次病程记录生成程序

本项目面向教学比赛，不用于真实诊疗。当前研究版本 `0.4.9-prompt-v2-research-modules` 在保留 Baseline 与 Evidence Pipeline 的基础上，提供自动输入协议识别、对比式临床规划、问题条件化证据写作、病例内证据检索、非阻断式风险核验和反馈驱动的单轮局部修订，并加入默认关闭、带完整执行审计的轻量临床图重排、一次性反证代理、五阶段 Prompt v2、临床写作辅助资源和单病例 Shared Clinical Memory；正式 Schema、赛方模型参数和资源配额尚待公布。

## 运行

Python 3.11 兼容性开发。正式提交包在包根目录放置 `.env` 和 `input.json` 后，直接无参数启动：

```bash
python main.py
```

最新材料要求的两参数入口同样是 Profile 驱动入口，输入输出真正使用命令行路径：

```bash
python3.11 main.py --input input.json --output output.json
```

无参数模式严格读取包根目录 `.env` 中的 `MODEL_NAME` 和完整 `ENDPOINT`，读取 `input.json`，并原子生成 `output.json` 与 `run_log.json`。`ENDPOINT` 不会被拼接；正式客户端不读取 `DASHSCOPE_API_KEY`、不发送 Authorization，也不会回退开发或 Mock 服务。当前保守默认采用一次调用的 Baseline、`text` 响应格式、零重试，避免在赛方配额与结构化输出能力公布前擅自假设。`.env.example` 只展示字段，不包含真实地址或凭据；可选 `TIMEOUT_SECONDS` 和 `MAX_RETRIES` 是有界工程控制，后者最多 10 次且每次均计入预算。

正式算法由 [src/competition_profile.py](src/competition_profile.py) 的打包常量选择，不增加 `.env` 字段，也不要求赛方添加 CLI 参数。内置 `baseline_safe`、`clinical_d`、`clinical_d_agent`、`clinical_d_graph_agent` 四档；当前 `ACTIVE_COMPETITION_PROFILE="baseline_safe"`。只有官方确认至少 5 次 Qwen 调用、上下文/Token 预算及 BGE-M3 HTTP 协议，并通过固定病例复测后，才把打包常量切换到 `clinical_d_graph_agent`。正式 Profile 会在启动时校验其保留预算与实际阶段数一致；Clinical 档对仍未修复的 HIGH 风险设置最终阻断门。

本地完整链路必须显式设置开发开关；程序不会因为检测到百炼变量而自动切换：

```bash
export MEWTYPE_DEV_MODE=1
export DASHSCOPE_API_KEY='...'
export QWEN_BASE_URL='https://.../compatible-mode/v1'
export QWEN_MODEL='qwen-plus'

python3.11 main.py --input input.json --output output.json
```

该命令自动选择 `development_clinical_d_graph_agent`：contrastive、Verifier/Reviser on、Agent always、PPR、阶段化 `json_schema`，并使用明确标记为开发测试的确定性 Mock Embedding。Mock 不是 BGE-M3，不会在 competition Profile 中启用。开发 Profile 重跑前会把已存在的输出和 `run_log.json` 复制为带 UTC 时间与 run_id 的相邻 backup，避免覆盖首次 A 版结果；正式评测仍按指定输出路径原子写入。

带参数的旧 CLI 继续用于开发实验。其相对输入、输出、配置和日志路径按**启动命令的当前工作目录**解析；提示词按程序目录定位：

```bash
python main.py --input input.json --output output.json \
  --mode competition --config config/competition.json --pipeline baseline
```

显式 CLI 的正式配置缺失时程序以退出码 2 明确失败，绝不回退到本地模型。配置中的 `endpoint` 必须是赛方提供的完整地址；当前版本不发送认证头，也不猜测认证方式。未知官方配额没有默认值，多阶段实验的 `max_calls_per_case` 必须显式配置。无参数正式档的预算仅覆盖一次 Baseline 及显式设置的有限重试。

### 阿里云百炼开发模式

`dev_api` 只用于本地开发验证，与 `competition` 完全隔离。它只读取以下环境变量，不读取配置文件，也不记录密钥：

```bash
export DASHSCOPE_API_KEY='...'
export QWEN_BASE_URL='https://.../compatible-mode/v1'
export QWEN_MODEL='qwen-plus'

python main.py --input test_input.json --output baseline_output.json \
  --log baseline_run_log.json --mode dev_api --pipeline baseline
```

`QWEN_BASE_URL` 可以是 OpenAI 兼容服务的 base URL；程序确定性追加 `/chat/completions`。若变量本身已经以 `/chat/completions` 结尾则不重复追加。缺少任一变量时明确失败，不回退 mock、local 或 competition。开发接口使用 Bearer 认证仅限 `dev_api`；competition 客户端继续不发送认证头。

Baseline 默认仍使用 `text`，以免把开发服务能力误认为赛事接口能力。可用下列参数显式测试百炼结构化输出；该测试不配置回退，服务拒绝 `json_schema` 时会返回 `RESPONSE_FORMAT_REJECTED`，不会静默改用 text 或再次调用：

```bash
python3.11 main.py --input test_input.json --output baseline_output.json \
  --log baseline_run_log.json --mode dev_api --pipeline baseline \
  --dev-response-format json_schema
```

只有确认输入为允许保存原文的虚构开发病例时，才可显式将原始模型响应写到独立文件：

```bash
python3.11 main.py --input test_input.json --output baseline_output.json \
  --log baseline_run_log.json --mode dev_api --pipeline baseline \
  --dev-response-format json_schema \
  --debug-raw-response baseline_raw_response.json \
  --confirm-fictional-input
```

普通运行日志只记录模型标识、阶段、请求格式、调用/JSON解析状态、解析错误位置、Token、调用数、耗时和错误，不复制病例或原始回答。调试文件可能包含病例内容，默认关闭，且 `competition`、Evidence Pipeline 和未确认虚构输入时均拒绝启用；经确认的虚构 `clinical_pipeline` 病例也可用同一组调试参数定位各阶段原始响应。

离线 mock 必须显式选择并提供按调用顺序排列的响应脚本：

```bash
python main.py --input input.json --output output.json \
  --mode mock --mock-responses tests/fixtures/responses.json
```

日志会标记 `development_label=true` 和 `model=mock`，不能把它当作真实模型结果。

本机 MLX 直接调用仅供开发，不是正式依赖：复制 `config/local.mlx.example.json`，把 `model` 改为已存在的本地快照路径，然后使用 `--mode local`。该适配器当前只验证 `text`；不会因本地支持某能力而假定赛事接口支持。提交包不包含 MLX、权重、缓存或本地路径配置。

## 输入与输出

入口默认使用 `--input-format auto` 严格识别两种格式：正式比赛 JSON 数组，或中文嵌套练习病历（单对象或同类对象数组）。正式数组仍直接进入原有协议解析器：`case_id`、`record_cutoff`、病例顺序、`source_id`、`field_path` 和内容不会被重新生成。顶层损坏、缺失/重复 `case_id` 是任务级错误；截止点或来源无效仍按原协议形成单例失败。未知结构、单个正式 Case 对象、混合协议数组以及带有部分正式字段的对象不会被猜测转成练习格式。

练习病历可以直接作为 `--input`，无需先生成中间文件：

```bash
python3.11 main.py \
  --input "information/临床智录_练习病历数据集JSON/03_老年病科_多病共存TIA待排.json" \
  --output information/results/03/auto_output.json \
  --mode dev_api --pipeline clinical_pipeline
```

练习格式的每例必须含顶层 `入院记录`，可附带合法显式 `case_id`；`入院记录` 包含对象型 `一般项目` 和至少一个可识别临床章节。程序复用唯一的展平实现，在内存中确定性生成 `P0001...`，保留 `field_path`、原文和 `source_id_origin=program_derived`，不写转换中间文件。练习数组无显式 ID 时按输入顺序生成 `PRACTICE001...`。为兼容既有单文件工作流，文件名前导序号如 `03_` 仍可得到 `PRACTICE03`；改名为普通 `input.json` 时，则对排除姓名、证件、住院号、电话、住址、工作单位等直接身份字段后的原始结构生成稳定 SHA-256 短标识。科室、病种文件名和直接身份信息均不进入自动 case_id。

`record_cutoff` 优先读取 `入院记录.一般项目.记录时间`，在验证日历时间后把 U+2011 等 Unicode 连字符规范成 ASCII `-`；原始时间同时保留在运行日志的 `input_adapter.case_mappings` 审计字段中，不使用系统当前时间。可按需覆盖自动 case_id；病历无可靠记录时间时必须显式提供截止点：

```bash
python3.11 main.py --input practice.json --output output.json \
  --input-format practice --practice-case-id PRACTICE_CUSTOM \
  --practice-record-cutoff '2026-05-08 14:25' \
  --mode dev_api --pipeline clinical_pipeline
```

可用 `--input-format competition|practice` 强制协议并获得更早、更明确的格式错误。练习专用覆盖参数不能用于正式数组。

模型调用前可使用只读诊断工具检查格式、病例数、ID、来源数、截止点和内部 Schema；它不打印病例正文，也不写文件：

```bash
python3.11 tools/check_input.py --input input.json
```

旧离线转换工具继续保留用于调试和固定输入快照：

```bash
python tools/adapt_practice.py --input practice.json --output adapted.json \
  --case-id PRACTICE01 --record-cutoff '2026-05-03 20:30'
```

工具与自动入口共用同一套展平逻辑。自动入口和离线工具都不会修改原始 `information/`；自动入口只在经过格式验证后把练习记录时间作为资料截止点。

输出为旧规范数组。成功包含 `case_features`、`diagnostic_discussion`、`plan`、`evidence`、`review_items`；失败保留位置和 `case_id`。写入使用输出目录内临时文件并原子替换，空输入写 `[]`。

## 流程

- `baseline`：原始资料 → 一次模型生成 → 格式/来源/有限事实检查 → 输出。
- `evidence_pipeline`：原始资料 → 事实整理 → 原文+事实约束生成 → 独立核验 → 输出。
- `clinical_pipeline`：确定性事实编号 → 临床问题规划 → 可选病例内图重排/一次性反证审查 → 证据约束写作 → 可选原子事实核验 → 必要时最多一次局部修订 → 确定性复核 → 输出。

事实结构分别表达原文、来源、字段路径、事实内容、时间、数值、单位、肯定/否定/不确定/未提供、完成/进行中/计划及更正关系。原始 sources 始终随生成和核验传入，事实表不是唯一输入。该流程尚未通过同条件医学实验，不能宣称优于 baseline。

### 可选 Dense 证据检索

Dense 检索只允许显式用于 `evidence_pipeline` 或 `clinical_pipeline`，Baseline 路径完全不调用它。检索器将单病例来源切成完整陈述或受控长度片段，保留 `case_id/source_id/field_path/time`，批量编码来源，并缓存病例内来源向量和重复查询。它把每条生成 evidence 的 Top-K 候选交给独立核验器，同时标记明显的否定、时间及计划/完成状态冲突。

向量相似度只用于定位候选，既不代表陈述正确，也不会自动覆盖模型声明的 `source_ids`。候选不足时程序向 `review_items` 增加待核提示，现有结构、来源编号、数值、否定和状态校验保持启用。输入协议目前没有逐来源可用时间字段，因此检索严格限定在当前 `Case` 的 sources；调用方仍须保证这些 sources 已按 `record_cutoff` 截断，此限制待正式 Schema 提供来源时间后在适配层加强。

离线 mock 示例不联网，也不代表 BGE-M3 效果：

```bash
PYTHONPATH=. python3 examples/retrieval_mock_demo.py
```

本地 BGE-M3 仅供获准的开发验证。复制 `config/retrieval.local.example.json`，将 `model` 指向已下载的本地路径；默认 `allow_download=false`，程序不会下载模型。`FlagEmbedding` 是开发环境可选依赖，未写入正式 `requirements.txt`。启用命令示例：

```bash
python3 main.py --input input.json --output output.json \
  --mode local --config config/local.mlx.json --pipeline evidence_pipeline \
  --retrieval-config config/retrieval.local.json
```

`competition` 模式禁止 `mock` 和 `local_bge` 检索配置，只接受 `official`。当前 `OfficialEmbeddingClient.embed()` 是明确失败的协议边界；赛方公布后只在 `src/embedding_client.py` 中替换该方法并补充配置解析、认证、超时、重试和配额测试，不改病例处理、检索、核验或输出代码。不能因本地支持 Dense/Sparse/ColBERT 而假定赛方具备后两者。

三组 A/B/C 的冻结条件、标注与指标定义见 `docs/retrieval_experiment.md`。

### 可选 Graph-Guided Retrieval 与反证代理

两项研究功能均默认关闭，Baseline 不经过这些代码。`--graph-retrieval one_hop|ppr` 在当前病例内构建 Source、Fact、时间、临床问题、鉴别与计划目标的异构图，并对已有 Dense 候选重排；它必须同时提供 `--retrieval-config`。图分数不会自动覆盖最终 `source_ids`，现有数值、否定、时间和状态检查保持启用。

`--clinical-agents adaptive` 只在透明复杂度评分达到阈值时增加一次反证审查；`always` 仅用于对照。代理使用紧凑协议：`reviewed_problem_ids` 必须完整覆盖 Planner 问题，`issues` 只返回高价值反证、方向含混 Fact、关键缺失或确定性过强，不再逐问题输出长篇 summary。输入使用带 source_id 的原文 Fact 片段和精简问题视图，不重复整份 sources/plan。代理不修改 Facts/Plan；截断或 Schema 失败会记录 `completed=false`、finish reason 和覆盖错误后回到 Writer，不进入自由讨论循环。启用任一代理模式时，每例调用预算会预留一次额外调用。

Graph 模式下执行顺序为：Dense 候选 → Clinical Graph → one-hop/PPR 重排 → 紧凑检索候选交给 Counter-Evidence Agent → 同一组候选交给 Writer。Agent 和 Writer 都被明确告知 Dense/PPR 分数不是事实证明。病例日志的 `clinical.graph_retrieval` 记录 `graph_requested`、`graph_executed`、实际模式、fallback 原因、Embedding provider、种子/候选/重排 Fact ID、边类型、最终 Source ID、排序是否变化和耗时；Graph 未执行不会仅以 `enabled=true` 冒充成功。

离线查看现有转换病例的 Facts 和图结构（不调用 Qwen/BGE）：

```bash
python3.11 tools/inspect_clinical_graph.py \
  --input "information/临床智录_练习病历数据集JSON/03_老年病科_多病共存TIA待排.json" \
  --output /tmp/practice03_graph_audit.json
```

离线验收 PRACTICE03 的真实图构建与 PPR 重排代码（使用确定性 Mock 向量，不能冒充 BGE-M3）：

```bash
python3.11 tools/smoke_graph_retrieval.py \
  --output information/results/03/graph_ppr_smoke.json
```

使用本地/获准 embedding 做一跳图重排并按复杂度触发反证代理：

```bash
python3.11 main.py --input information/converted/01_input.json \
  --output graph_agent_output.json --log graph_agent_run_log.json \
  --mode dev_api --pipeline clinical_pipeline \
  --clinical-verification off --clinical-revision off \
  --retrieval-config config/retrieval.local.json \
  --graph-retrieval one_hop --clinical-agents adaptive \
  --clinical-response-format json_schema
```

若 `config/retrieval.local.json` 指向本地 BGE，开发机必须已具备依赖和权重；程序不会下载。若只想验证控制流，可用 `provider=mock`，但其哈希向量不代表 BGE 语义效果。PRACTICE03 的离线验收、真实 A/B 命令和审计字段见 `docs/d_graph_agent_integration_validation.md`；算法依据、固定评分、Agent Schema、成本和实验矩阵见 `docs/graph_rag_multi_agent_design.md`。

### Clinical Pipeline v0.4.4-research

#### 独立研究开关：Prompt v2、写作知识与共享记忆

当前稳定 `contrastive` 和正式 `baseline_safe` 均未改变。五阶段候选 Prompt 位于 `prompts/contrastive_v2/`，只能通过实验参数显式启用：

```bash
--clinical-prompt-profile contrastive_v2
```

Planner、Counter-Evidence Agent、Writer、Verifier、Reviser 分别加载同目录独立 Prompt，但继续使用既有 `PLANNER_SCHEMA`、`COUNTER_EVIDENCE_SCHEMA`、`MEDICAL_SCHEMA`、`VERIFIER_SCHEMA` 和 `REVISER_SCHEMA`。日志的 `prompt_manifest` 记录实际文件哈希，`prompt_audit` 记录每阶段 system/payload 字符数及准备耗时；真实 API 的 Token 和调用耗时仍由 `stage_calls` 分阶段记录。

Writer 写作知识默认关闭，可单独选择规则卡，或规则卡加原创合成示例的风格提要：

```bash
--clinical-writing-guidance rules
--clinical-writing-guidance rules_examples
```

资源只进入 Writer 的 `writing_style_context`，带有 `NOT_PATIENT_EVIDENCE` 标识。合成示例检索只返回 `style_takeaways`，不把合成病例正文、数值、药物、时间或 source_id 注入当前患者。Verifier/Reviser 不读取这些资源。

单病例 Shared Clinical Memory 同样默认关闭：

```bash
--shared-clinical-memory on
```

启用后，Planner 写入基于 Fact ID 的决策卡，Graph/PPR 标注候选 Fact，Counter Agent 增量标注问题，Writer 读取紧凑卡片后仍须回查原始 Sources。对象仅在当前病例处理期间存在，不持久化、不跨病例，也不增加 Qwen 调用；Verifier/Reviser 始终直接使用原始 Sources。三个研究功能应分别做消融，不应同时打开后把差异归因给单一模块。完整接入与实验命令见 `docs/prompt_v2_shared_memory_writing_knowledge.md`。

Clinical Planner 先输出内部问题表，包括优先级、来源属性、支持/反对事实、鉴别诊断、区分所需资料、计划目标和检索查询。提示词要求每个主要问题优先保留2–3个有依据的鉴别方向；只有原始资料明确否定或确有不支持事实时才列为反对证据。所有事实使用程序确定生成的 `F001...`，不存在的事实引用会使单例失败。初步诊断必须标记为 `documented_problem` 或 `mixed`，不能直接视为已证实结论。

Planner 后由 Python 构建 `problem_evidence_context`，确定性地解析 Fact ID，并按问题组织支持证据、限制证据、各鉴别方向的支持/限制/待区分资料和 plan goals。Writer 明确优先使用这一视图，再回到原始 sources 核对；Planner 仍不是患者事实来源。显著性排序提高问题直接关联事实、数值/单位、明确阴性、更正、时序、状态、异常检查、暴露和治疗史的优先级，降低无关人口学字段优先级。排序只改变组织顺序，不修改事实文本。

启用检索时，每个问题的查询会确定性补入已引用的患者事实、明确限制和待区分资料，再只检索当前病例 sources，形成按 `problem_id` 归组的 Evidence Bundle。Contrastive Writer 接收原始 sources、问题表和去重后的问题证据视图：事实正文只在 Fact-ID catalog 中保存一次，问题/鉴别关系只引用 Fact ID，检索候选按问题归组；检索结果不是唯一输入，也不自动成为最终 evidence。Legacy 档位继续使用原有 facts/evidence_bundles 载荷，避免改变归档提示词的输入契约。

若同一鉴别诊断把同一 Fact ID 同时列为支持和反对，程序会先确认该 ID 合法，再从两个方向同时移除歧义引用，并在 `missing_evidence` 和日志中记录“方向需复核”；不会擅自替模型判断方向，也不会改写原始 facts。规范化后仍无效时，仅对 `INVALID_CLINICAL_PLAN` 或 `MODEL_INVALID_JSON` 启用有限回退：预算足够则最多尝试一次 Legacy Planner，否则用剩余一次调用执行 Baseline；日志中的 `algorithm_mode`/`planner_fallback` 如实记录实际路径，预算不足明确失败。

`--clinical-prompt-profile` 支持三档；正常有效规划不会增加调用次数：`legacy` 使用归档 v0.4 提示词；`contrastive` 是默认的强化规划/写作；`contrastive_fewshot` 从合成示例库按病例关键词为 Planner 和 Writer 各选择最多一个示例。有限回退只在规划结果无效且预算足够时使用已有病例调用预算。日志记录 profile、实际算法模式、示例 ID、规范化摘要及每阶段输入字符数。Few-shot 仅用于实验，不代表质量已经提升。

Writer 之后，程序对五个输出部分拆分可定位的原子陈述，并为正文记录字符区间、为 evidence/review_items 记录项目下标。确定性规则先检查未知写成否定、频次缺失、事实反转、诊断确定程度、更正、时间和计划状态；可选 Qwen Verifier 使用紧凑协议，只返回完整的 `checked_claim_ids` 和真正需要关注的 `issues`，不再为每条正确 Claim 重复 verdict/rationale。程序要求 checked ID 无重复且完整覆盖全部 Claim，否则明确记录 `INCOMPLETE_CLAIM_COVERAGE`，绝不把未提及视为通过。

HIGH issue 按 Claim ID 合并成唯一 Patch 目标后，才可能触发第四次 Local Reviser；MEDIUM 不使整例失败，进入待核事项；LOW 不触发额外调用。Reviser 只能返回 `replace/remove` Claim Patch 和临床待核事项，不能重写整个字段；程序使用字符区间或列表下标精确应用、验证结构和来源，并重新执行确定性高风险检查。若修订未启用或模型修订失败，确定性保守替换会单独标记为 `deterministic_fallback_applied`，不会把 `revision_applied` 伪装为模型修订成功。模型/JSON/API 错误只进入运行日志，禁止写入面向医生的 review_items。详细设计与旧 D 实测诊断见 `docs/clinical_verifier_revision_optimization.md`。

消融运行方式：

```bash
# B：Planner + Writer，共2次Qwen；无Embedding、无核验/修订
python3.11 main.py --input input.json --output b.json --mode dev_api \
  --pipeline clinical_pipeline --clinical-verification off --clinical-revision off \
  --clinical-prompt-profile legacy

# C：Contrastive Planner + Problem-Conditioned Writer；不使用核验
python3.11 main.py --input input.json --output c.json --mode dev_api \
  --pipeline clinical_pipeline --clinical-verification off --clinical-revision off \
  --clinical-prompt-profile contrastive

# D：C + 原子事实核验 + 必要时一次局部修订
python3.11 main.py --input input.json --output d.json --mode dev_api \
  --pipeline clinical_pipeline --clinical-verification on --clinical-revision on \
  --clinical-prompt-profile contrastive

# E：D + 已获准且实际可用的BGE-M3
python3.11 main.py --input input.json --output e.json --mode dev_api \
  --pipeline clinical_pipeline --clinical-verification on --clinical-revision on \
  --clinical-prompt-profile contrastive \
  --retrieval-config config/retrieval.local.json
```

不提供 `--retrieval-config` 时 B/D/E 均可运行，用于检索消融。Planner、Counter Evidence、Medical Record、Verification 和 Revision 分别使用 `PLANNER_SCHEMA`、`COUNTER_EVIDENCE_SCHEMA`、`MEDICAL_SCHEMA`、`VERIFIER_SCHEMA` 和 `REVISER_SCHEMA`，解析后仍执行程序侧字段、枚举和引用校验。默认继续采用配置中的格式；可用 `--clinical-response-format json_schema` 显式要求所有阶段使用各自 Schema，也可通过 `--planner-response-format`、`--counter-evidence-response-format`、`--writer-response-format`、`--verifier-response-format`、`--reviser-response-format` 单独覆盖为 `text/json_object/json_schema`。服务不支持时直接报告 `RESPONSE_FORMAT_REJECTED`，不会自动假装支持；仅配置文件中显式声明的 fallback 才可能增加调用。competition 配置的 `max_calls_per_case` 必须覆盖所选消融的2/3/4次预算，启用反证代理再增加1次，且不能因百炼支持而假定赛方内网支持。

虚构 `TEST001` 的分阶段结构化输出调试示例：

```bash
python3.11 main.py --input test_input.json --output clinical_output.json \
  --log clinical_run_log.json --mode dev_api --pipeline clinical_pipeline \
  --clinical-verification on --clinical-revision on \
  --clinical-response-format json_schema \
  --debug-raw-response clinical_raw_response.json --confirm-fictional-input
```

若服务明确拒绝某一阶段的 `json_schema`，先保留拒绝日志，再显式把该阶段改为已验证的 `json_object`，例如 `--verifier-response-format json_object`；程序不做静默回退。

最终 evidence 目标为3–5条且绝不超过5条。程序会先移除结构或来源无效项、合并确定性重复项，再按数值/否定/更正/诊断状态/时间价值及来源覆盖选择；不足3条时按实际保留，不补造。无效项会在 review_items 披露。`--clinical-evidence-limit` 只能设为1–5，该限制不删除内部 facts、claims 或检索候选。

`tools/eval_clinical_pipeline.py` 接受包含 A–E 输入、输出、日志和可选人工标注路径的 manifest，自动汇总三段完整率、正文字符数、结构、来源、调用、Token 与耗时；关键事实错误、事实支持、诊断依据覆盖、反对/限制依据准确性、计划对应、局部修订及 Recall@K/MRR 只有提供人工标注时才计算，不使用模型自评冒充真值。格式见 `docs/clinical_experiment.md`。

`response_format` 可设为 `text`、`json_object`、`json_schema`。能力回退只按 `fallback_response_formats` 显式执行，每次回退计入预算；MLX 直接适配器不支持的格式直接报错。分阶段 Schema 已在百炼开发适配器中具备测试入口，但赛事服务能力仍须独立验证。

## 校验能力边界

程序分开检查结构、来源编号、引用中数值/单位的存在性、日期等价格式、否定支持和“计划写成已完成”等风险。`2026.05` 与 `2026年5月` 可规范为同一月份，但不同年份或月份仍报告冲突；空格、`μ/µ` 和常见 Unicode 连字符会规范化，而 `10`、`11.8`、`14.2` 绝不视为等价。数值检查绑定 evidence 陈述与其引用来源，不用全文数字集合冒充医学核验。若 kind=fact 的短来源 Evidence 含不支持数值，程序只允许用所引短来源原文做一次确定性替换；推断阈值或无法唯一确认的数值保持错误并使最终验证失败。Writer 后和最终输出前分别记录 Evidence 校验/修复摘要。规则仍无法判断所有语义支持、临床合理性和隐含时序，风险项需要人工或专业审核。修订幅度日志仅记录字段、长度、字符变化估计、SequenceMatcher 相似度和文本哈希，不在普通日志复制完整病例。

## 日志与复现

默认在输出目录写 `run_log.json`，记录 run_id、程序/提示词版本、所用 Prompt 文件 SHA-256、算法 Profile、环境、实际模型标识、非敏感参数、病例标识、每例状态、调用次数、Token usage、各阶段耗时、Graph/Agent 审计、最终验证状态、错误和检查结果。普通日志不复制完整病例，不记录密钥。固定种子只能控制客户端可控部分，不能承诺远程模型逐字确定复现。

## 测试与打包

```bash
python3 -m unittest discover -s tests -v
python3 tools/build_package.py --team 团队名称 --version v0.4.9-prompt-v2-research-modules
```

打包脚本生成一层团队目录，仅包含运行文件、源码、提示词和资源清单，排除 `information/`、配置、凭据、缓存、虚拟环境、测试输出和本地模型。

## 新旧通知差异及暂定处理

| 项目 | 旧规范 | 2026-11-25材料通知 | 当前处理 |
|---|---|---|---|
| 启动 | `python main.py` 无参数 | `python main.py --input ... --output ...` | 采用最新命令 |
| 依赖文件 | `requires.txt` | `requirements.txt` | 采用最新名称 |
| ZIP结构 | 根目录直接是文件 | ZIP内一层 `team_name_v1/` | 采用最新一层目录 |
| 必需资源文件 | `.env.example` | `resource_manifest.json`、可选 resources | 采用最新清单；配置单独提供 |
| Python | 明确3.11 | 版本/CPU/内存待正式说明 | 以3.11开发，标记待确认 |
| 输入输出 | 旧数组及 `output.json` | Schema 将另发 | 协议层暂实现旧版，可替换 |
| 结果文件 | 前文 `results.json`，附件 `output.json` | 由命令参数指定 | 尊重 `--output`；待正式 Schema 确认 |
| 提交材料 | 算法包、结果/日志包、≤10页说明、声明 | 邮件提交完整算法 ZIP，README和资源清单 | 本包只构建算法 ZIP；其他材料是否仍需提交待确认 |

## 待确认

赛方正式 Schema、Python/资源限制、Qwen/BGE-M3 完整地址与模型名、Embedding 请求/响应格式及 1024 维约束、认证、批大小、生成参数、`response_format` 支持、每例/全任务预算、并发、超时、重试及计费规则、日志 Schema、额外说明和声明材料要求。BGE-M3 默认关闭且不是正式依赖；只有确认必要、获准且完成版本登记后才启用 official 适配器。
