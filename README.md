# 循证智录：首次病程记录生成程序

本项目面向教学比赛，不用于真实诊疗。当前版本 `0.3.1` 实现旧赛题输出协议、最新通知的命令行入口、与正式比赛接口隔离的百炼开发适配器，以及可关闭的病例内 Dense 证据检索实验模块；正式 Schema、赛方模型参数和资源配额尚待公布。

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

普通运行日志只记录模型标识、请求过的格式、实际成功格式、Token、调用数、耗时和错误，不复制病例或原始回答。调试文件可能包含病例内容，默认关闭，且 `competition`、Evidence Pipeline 和未确认虚构输入时均拒绝启用。

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

事实结构分别表达原文、来源、字段路径、事实内容、时间、数值、单位、肯定/否定/不确定/未提供、完成/进行中/计划及更正关系。原始 sources 始终随生成和核验传入，事实表不是唯一输入。该流程尚未通过同条件医学实验，不能宣称优于 baseline。

### 可选 Dense 证据检索

Dense 检索只允许显式用于 `evidence_pipeline`，Baseline 路径完全不调用它。检索器将单病例来源切成完整陈述或受控长度片段，保留 `case_id/source_id/field_path/time`，批量编码来源，并缓存病例内来源向量和重复查询。它把每条生成 evidence 的 Top-K 候选交给独立核验器，同时标记明显的否定、时间及计划/完成状态冲突。

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

`response_format` 可设为 `text`、`json_object`、`json_schema`。能力回退只按 `fallback_response_formats` 显式执行，每次回退计入预算；MLX 直接适配器不支持的格式直接报错。当前 `json_schema` 仅是客户端扩展位置，在赛方 Schema 和服务能力确认前不要启用。

## 校验能力边界

程序分开检查结构、来源编号、引用中数值/单位的存在性、否定支持和“计划写成已完成”等风险。数值检查绑定 evidence 陈述与其引用来源，不用全文数字集合冒充医学核验；但规则仍无法判断所有语义支持、临床合理性和隐含时序，警告项需要人工/专业审核。核验失败时不自动补造正文。

## 日志与复现

默认在输出目录写 `run_log.json`，记录 run_id、程序/提示词版本、环境、实际模型标识、非敏感参数、每例状态、调用次数、Token usage、耗时、错误和检查结果。普通日志不复制完整病例，不记录密钥。固定种子只能控制客户端可控部分，不能承诺远程模型逐字确定复现。

## 测试与打包

```bash
python3 -m unittest discover -s tests -v
python3 tools/build_package.py --team 团队名称 --version v0.3.1
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
