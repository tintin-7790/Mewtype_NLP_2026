# 循证智录：首次病程记录生成程序

本项目面向教学比赛，不用于真实诊疗。当前版本 `0.4.2` 在保留 Baseline 与 Evidence Pipeline 的基础上，提供问题导向神经符号生成、病例内证据检索、非阻断式风险核验和反馈驱动的单轮局部修订；正式 Schema、赛方模型参数和资源配额尚待公布。

## 运行

Python 3.11 兼容性开发；最终 Linux/Python 版本待赛方确认。相对输入、输出、配置和日志路径均按**启动命令的当前工作目录**解析；提示词按程序目录定位。

```bash
python main.py --input input.json --output output.json \
  --mode competition --config config/competition.json --pipeline baseline
```

正式配置缺失时程序以退出码 2 明确失败，绝不回退到本地模型。配置中的 `endpoint` 必须是赛方提供的完整地址；当前版本不发送认证头，也不猜测认证方式，待官方要求明确后再实现。未知官方配额没有默认值，`max_calls_per_case` 必须显式配置。

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

正式入口严格接受旧规范 JSON 数组：`case_id`、`record_cutoff`、`sources[]`，每个来源含 `source_id` 和 `content`。顶层损坏、非数组、缺失/重复 `case_id` 是任务级错误；截止点或来源无效是单例失败。未知协议不会被猜测接受。

练习中文嵌套 JSON 使用独立工具，必须显式给出病例标识和截止点：

```bash
python tools/adapt_practice.py --input practice.json --output adapted.json \
  --case-id PRACTICE01 --record-cutoff '2026-05-03 20:30'
```

工具确定性生成 `P0001...`，并保留 `field_path`、原文和 `source_id_origin=program_derived`。文件名不进入医学证据，记录时间不会自动当作截止点。原始 `information/` 不会被修改。

输出为旧规范数组。成功包含 `case_features`、`diagnostic_discussion`、`plan`、`evidence`、`review_items`；失败保留位置和 `case_id`。写入使用输出目录内临时文件并原子替换，空输入写 `[]`。

## 流程

- `baseline`：原始资料 → 一次模型生成 → 格式/来源/有限事实检查 → 输出。
- `evidence_pipeline`：原始资料 → 事实整理 → 原文+事实约束生成 → 独立核验 → 输出。
- `clinical_pipeline`：确定性事实编号 → 临床问题规划 → 可选问题级 Dense 检索 → 证据约束写作 → 可选原子事实核验 → 必要时最多一次局部修订 → 确定性复核 → 输出。

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

### Clinical Pipeline v0.4.2

Clinical Planner 先输出内部问题表，包括优先级、来源属性、支持/反对事实、鉴别诊断、缺失证据、计划目标和检索查询。所有事实使用程序确定生成的 `F001...`，不存在的事实引用会使单例失败。初步诊断必须标记为 `documented_problem` 或 `mixed`，不能直接视为已证实结论。

启用检索时，每个问题的查询只检索当前病例 sources，形成包含来源、字段路径、原文、相似度、时间、断言和状态冲突提示的 Evidence Bundle。Writer 同时接收原始 sources、事实、问题表和证据包；检索结果不是唯一输入，也不自动成为最终 evidence。

Writer 之后，程序对五个输出部分拆分原子陈述。确定性规则先检查未知写成否定、频次缺失、事实反转、诊断确定程度、更正、时间和计划状态；可选 Qwen Verifier 再批量返回逐 claim verdict 与结构化问题。HIGH 优先触发第四次局部修订；MEDIUM 不使整例失败，进入待核事项；LOW 不触发额外调用。无来源的 GROUNDED/PARTIALLY_GROUNDED 推断不会伪造来源，而会降级为 REQUIRES_CLINICIAN_REVIEW。若修订未启用、预算不足或修订响应无效，程序对已精确定位的 HIGH 陈述进行保守替换并在 review_items 披露，内部标记 `fallback_disclosed`，不伪装为完全核验通过。未定位字段保持不变；修订最多一轮，之后重新执行确定性规则和原有 validation。

消融运行方式：

```bash
# B：Planner + Writer，共2次Qwen；无Embedding、无核验/修订
python3.11 main.py --input input.json --output b.json --mode dev_api \
  --pipeline clinical_pipeline --clinical-verification off --clinical-revision off

# C：B + 问题级Dense检索
python3.11 main.py --input input.json --output c.json --mode dev_api \
  --pipeline clinical_pipeline --clinical-verification off --clinical-revision off \
  --retrieval-config config/retrieval.local.json

# D：C + 原子事实核验；不调用模型修订，HIGH 使用保守替换并披露
python3.11 main.py --input input.json --output d.json --mode dev_api \
  --pipeline clinical_pipeline --clinical-verification on --clinical-revision off \
  --retrieval-config config/retrieval.local.json

# E：完整流程；正常3次Qwen，必要时第4次局部修订
python3.11 main.py --input input.json --output e.json --mode dev_api \
  --pipeline clinical_pipeline --clinical-verification on --clinical-revision on \
  --retrieval-config config/retrieval.local.json
```

不提供 `--retrieval-config` 时 B/D/E 均可运行，用于检索消融。Planner、Medical Record、Verification 和 Revision 分别使用 `PLANNER_SCHEMA`、`MEDICAL_SCHEMA`、`VERIFIER_SCHEMA` 和 `REVISER_SCHEMA`，解析后仍执行程序侧字段、枚举和引用校验。默认继续采用配置中的格式；可用 `--clinical-response-format json_schema` 显式要求所有阶段使用各自 Schema，也可通过 `--planner-response-format`、`--writer-response-format`、`--verifier-response-format`、`--reviser-response-format` 单独覆盖为 `text/json_object/json_schema`。服务不支持时直接报告 `RESPONSE_FORMAT_REJECTED`，不会自动假装支持；仅配置文件中显式声明的 fallback 才可能增加调用。competition 配置的 `max_calls_per_case` 必须覆盖所选消融的2/3/4次预算，且不能因百炼支持而假定赛方内网支持。

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

`tools/eval_clinical_pipeline.py` 接受包含 A–E 输入、输出、日志和可选人工标注路径的 manifest，自动汇总结构、来源、调用、Token 与耗时；事实支持、诊断依据覆盖、计划对应、局部修订及 Recall@K/MRR 只有提供人工标注时才计算，不使用模型自评冒充真值。格式见 `docs/clinical_experiment.md`。

`response_format` 可设为 `text`、`json_object`、`json_schema`。能力回退只按 `fallback_response_formats` 显式执行，每次回退计入预算；MLX 直接适配器不支持的格式直接报错。分阶段 Schema 已在百炼开发适配器中具备测试入口，但赛事服务能力仍须独立验证。

## 校验能力边界

程序分开检查结构、来源编号、引用中数值/单位的存在性、日期等价格式、否定支持和“计划写成已完成”等风险。`2026.05` 与 `2026年5月` 可规范为同一月份，但不同年份或月份仍报告冲突。数值检查绑定 evidence 陈述与其引用来源，不用全文数字集合冒充医学核验；规则仍无法判断所有语义支持、临床合理性和隐含时序，风险项需要人工或专业审核。修订幅度日志仅记录字段、长度、字符变化估计、SequenceMatcher 相似度和文本哈希，不在普通日志复制完整病例。

## 日志与复现

默认在输出目录写 `run_log.json`，记录 run_id、程序/提示词版本、环境、实际模型标识、非敏感参数、每例状态、调用次数、Token usage、耗时、错误和检查结果。普通日志不复制完整病例，不记录密钥。固定种子只能控制客户端可控部分，不能承诺远程模型逐字确定复现。

## 测试与打包

```bash
python3 -m unittest discover -s tests -v
python3 tools/build_package.py --team 团队名称 --version v0.4.2
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
