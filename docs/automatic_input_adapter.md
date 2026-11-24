# Automatic Input Adapter 技术报告

版本：`0.4.5-auto-input`

## 1. 目标与边界

Automatic Input Adapter 只解决输入边界的协议识别与练习数据转换。Clinical Planner、Graph Retrieval、Counter-Evidence Agent、Writer、Claim Verifier、Local Reviser 和最终输出协议均未改变。

程序默认执行：

```text
UTF-8 JSON → 严格格式检测 → 正式协议解析或练习内存适配 → 统一 Case → 原有 Pipeline
```

适配过程不调用 Qwen/BGE，不联网，不写转换中间文件，不根据患者姓名或身份证生成标识，也不使用系统当前时间补齐截止点。

## 2. 格式检测规则

`--input-format` 支持 `auto`、`competition`、`practice`，默认 `auto`。

| 输入 | auto 行为 |
|---|---|
| JSON 数组，包括空数组 | 作为正式协议交给原有 Case Parser |
| 仅含 `入院记录` 的对象，且含对象型 `一般项目` 与至少一个临床章节 | 作为练习格式 |
| 单个含 `case_id/record_cutoff/sources` 的对象 | 明确拒绝；正式协议当前要求数组 |
| 含部分正式字段的对象 | 明确拒绝；不尝试练习转换 |
| 弱匹配或未知对象 | `INPUT_FORMAT_UNKNOWN` 或 `PRACTICE_PROTOCOL_ERROR` |

显式 `competition` 始终要求顶层数组；显式 `practice` 始终执行完整练习结构校验。练习专用覆盖参数用于正式数组时返回 `PRACTICE_OVERRIDE_SCOPE_INVALID`。

## 3. 正式比赛输入

`src/input_adapter.py` 加载 JSON 后，把正式数组直接交给 `src/protocol.py::parse_standard_cases`。该函数由原 `load_standard_cases` 的解析主体提取而来，因此保留原有行为：

- 保留病例顺序、`case_id` 和 `record_cutoff`；
- 保留病例内来源顺序与原始 `source_id`；
- 不重编号、不展平、不改变 evidence 引用语义；
- 重复或缺失 `case_id` 仍是任务级错误；
- 截止点或来源错误仍保留为对应病例失败。

当前正式协议仍要求顶层数组；单个正式 Case 对象没有被擅自扩展为新协议。

## 4. 练习病历适配

`src/practice_adapter.py` 新增内存函数 `adapt_practice_data`。旧 `adapt_file` 和自动入口均调用它，因此 Sources 的展平顺序、字段路径、内容和编号只有一套实现。

### case_id

文件名以一至四位数字和下划线开头时，例如 `03_...json` 稳定映射为 `PRACTICE03`。文件名其余科室、病种或提示文字不进入来源或医学推理。改名为普通 `input.json` 时，对替换姓名、证件号、住院号、电话、住址、工作单位等直接身份字段后的原始结构计算 SHA-256，生成稳定的 `PRACTICE_AUTO_<12位摘要>`；也可用 `--practice-case-id` 显式覆盖。

### record_cutoff

默认读取 `入院记录.一般项目.记录时间`。程序执行 Unicode NFKC、常见 Unicode 连字符到 `-` 的映射和严格日历校验，仅接受：

- `YYYY-MM-DD HH:MM`
- `YYYY-MM-DD HH:MM:SS`

缺失或无效时明确失败；可用 `--practice-record-cutoff` 提供经过同样校验的值。程序不使用文件修改时间或系统当前时间。

运行日志保存原始时间、规范化时间、取值来源、病例 ID 来源和来源数量，用于审计，但不复制完整病历。原始记录时间本身也仍作为确定性展平来源存在。

### Sources

空值不生成来源；其余叶节点按原 JSON 顺序生成：

```json
{
  "source_id": "P0001",
  "field_path": "入院记录.一般项目.姓名",
  "content": "原始叶节点文本",
  "source_id_origin": "program_derived"
}
```

数组路径使用 `[index]`，对象或数组叶值按原转换器规则序列化。适配器不补写正常结果、时间、单位或诊断。

## 5. CLI 与审计

原命令保持有效：

```bash
python main.py --input input.json --output output.json
```

练习病例可直接运行：

```bash
python3.11 main.py \
  --input "information/临床智录_练习病历数据集JSON/03_老年病科_多病共存TIA待排.json" \
  --output information/results/03/auto_output.json \
  --mode dev_api --pipeline clinical_pipeline
```

普通运行日志新增 `input_adapter`，只记录检测格式、是否适配、映射来源、截止时间与来源数，不记录完整病例。正式输入只记录 `adapted=false` 和病例数。

离线 Graph 工具也使用同一输入边界，可直接审计 PRACTICE03：

```bash
python3.11 tools/inspect_clinical_graph.py \
  --input "information/临床智录_练习病历数据集JSON/03_老年病科_多病共存TIA待排.json" \
  --output information/results/03/auto_graph_audit.json
```

该审计只运行确定性 Facts 与病例内图构建，不调用模型，不代表医学判断。

## 6. 错误与安全策略

主要任务级错误包括：

- `INPUT_INVALID_JSON`：不是有效 UTF-8 JSON；
- `INPUT_FORMAT_UNKNOWN`：不属于任何受支持协议；
- `INPUT_PROTOCOL_ERROR`：正式协议顶层或字段形态错误；
- `PRACTICE_PROTOCOL_ERROR`：练习结构不完整或不严格；
- `PRACTICE_CASE_ID_INVALID`：显式 case_id 为空；无文件序号时默认使用脱敏稳定摘要；
- `PRACTICE_RECORD_CUTOFF_MISSING/INVALID`：记录时间缺失或无效；
- `PRACTICE_OVERRIDE_SCOPE_INVALID`：对正式输入使用练习覆盖参数。

所有错误均在模型调用前产生，不生成虚假患者来源。适配器不会修改原始输入，也不会把文件名的科室或病种当成证据。

## 7. 验证结果

专项测试覆盖正式单例数组、多病例、PRACTICE01/02/03、旧转换器一致性、来源编号、Unicode 日期、无效 JSON、缺失时间、未知格式、证据追溯、默认入口和无中间文件。

真实练习数据的确定性结果：

| 病例 | case_id | 规范化 record_cutoff | Sources |
|---|---|---|---:|
| 01 | PRACTICE01 | 2026-05-03 20:30 | 50 |
| 02 | PRACTICE02 | 2026-06-02 10:35 | 56 |
| 03 | PRACTICE03 | 2026-05-08 14:25 | 52 |

三例自动转换的 Sources 与旧 `adapt_file` 输出逐项一致。PRACTICE03 已完成离线 Graph 审计；该结果只证明输入转换、来源追溯和图构建可运行，不证明生成质量或医学正确性。
