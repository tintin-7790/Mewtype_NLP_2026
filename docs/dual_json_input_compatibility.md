# 双 JSON 输入协议兼容与回归验收

## 1. 范围与结论

本轮只修改输入边界、只读诊断工具、测试和说明文档。Clinical Planner、Graph/PPR、Counter-Evidence Agent、Writer、Verifier、Reviser、Prompt、正式输出 Schema 和 `ACTIVE_COMPETITION_PROFILE` 均未改变。

统一入口继续为：

```bash
python3.11 main.py --input input.json --output output.json
```

两种输入最终都进入同一 `Case(case_id, record_cutoff, sources)`，不会因输入格式选择不同生成算法。

## 2. 实际支持的输入

### 2.1 competition

顶层必须为 JSON 数组。空数组合法；每例由既有正式 Parser 校验 `case_id`、`record_cutoff` 和 `sources`。适配层不重新编号、不重排，也不改写 `source_id`、`field_path`、`content` 或 `source_id_origin`。缺失/重复 `case_id` 是任务级错误；单例截止点或来源错误保持既有失败语义。

### 2.2 practice

支持单个中文嵌套对象，也支持由同类对象组成的非空数组。每例必须包含对象型 `入院记录.一般项目` 和至少一个可识别临床章节，顶层只允许 `入院记录` 及可选的显式 `case_id`。

适配层复用 `src/practice_adapter.py` 的唯一展平逻辑，按原字段遍历顺序生成 `P0001...`，保留原始叶值、字段路径和 `source_id_origin=program_derived`。空字符串、空数组、空对象不会被改写成“正常”或否定事实。

病例 ID 规则：显式 CLI 覆盖优先；其次使用原始合法 `case_id`。练习数组其余病例按位置生成 `PRACTICE001...`。单文件继续兼容既有前导序号规则；普通 `input.json` 使用排除直接身份字段后的内容指纹，因此不依赖患者姓名等个人信息。

截止点来自 `入院记录.一般项目.记录时间`，支持 ASCII 及常见 Unicode 日期横线，验证为真实日历时间后规范化。不会使用系统当前时间；缺失时明确失败。原始记录时间只保存在适配审计中，原始输入不被改写。

## 3. 严格自动检测

- 空数组：competition（保持正式空任务语义）。
- 所有元素均含正式协议信号：competition，再交由正式 Parser 完整校验。
- 所有元素均为严格练习对象：practice。
- 正式与练习元素混合：`INPUT_FORMAT_MIXED`。
- 部分练习或部分正式元素夹杂未知元素：明确协议错误。
- 单个正式 Case 对象：拒绝；正式协议仍要求数组。
- 显式 `--input-format` 不会转换另一种协议。

检测只依据解析后的 JSON 结构，不搜索原始文本，也不依据文件名中的科室或病种。

## 4. 只读诊断工具

```bash
python3.11 tools/check_input.py --input input.json
```

输出仅含格式、病例数、case_id、来源数、截止点和 Schema 状态。它不调用 Qwen，不打印姓名或病例正文，不生成中间文件。返回码 0 表示全部通过；协议或 Schema 错误返回非零。

## 5. 练习病例验收

| 病例 | 结果 | case_id | cutoff | sources |
|---|---|---:|---:|---:|
| 01 | 通过 | PRACTICE01 | 2026-05-03 20:30 | 50 |
| 02 | 通过 | PRACTICE02 | 2026-06-02 10:35 | 56 |
| 03 | 通过 | PRACTICE03 | 2026-05-08 14:25 | 52 |
| 07 | 未通过输入解析 | — | — | — |

01–03 自动转换结果与旧 `adapt_file` 的 Sources 逐项一致，PRACTICE03 的 Source–Fact 图追溯测试通过。

第 07 份仓库原文件当前不是合法 JSON：解析器报告 `Extra data: line 67 column 1 (char 3660)`。文件第 66 行对象已结束，第 67 行仍有尾随 `]`。适配器按要求拒绝并未静默删除字符，也未修改原文件。因此 PCI 日期、用药依从性、患者自述肺部阴影和待排诊断的逐项来源验收目前被该原始资料错误阻塞；修正应由资料提供方完成并保留来源版本。

## 6. 回归覆盖

输入适配专项现为 28 项，覆盖正式单/多病例、练习单对象/数组、数组端到端输出顺序、严格检测、显式协议、原始来源保留、稳定编号、字段保真、Unicode 时间、缺失时间/来源、非法 JSON、混合数组、重复 ID、空字段、输入字节不变、Evidence/Graph 追溯、顺序、含中文/空格路径和类 Windows 路径字符、诊断工具隐私边界及第 07 份损坏输入。

完整回归为 `166/166` 通过；`python3.11 -m compileall -q main.py src tools tests` 与 `git diff --check` 均通过。所有测试均为离线 Mock/规则测试，没有调用真实 Qwen 或 Embedding。

## 7. 路径与重复运行

入口与诊断工具均使用 `pathlib.Path`，不自行按 `/` 或 `\\` 拆分路径；相对路径按当前工作目录解析，绝对路径直接使用。已覆盖中文、空格及反斜线字符路径。由于本轮运行主机为 macOS，真实 Windows Python 文件系统执行仍待 Windows 环境复验，不能把结构兼容测试表述为已完成跨平台实机认证。

输入只读；输出继续使用临时文件原子替换。重复运行不会生成转换中间病例。历史实验输出、日志与 `information/` 均未覆盖。

## 8. 已知边界

- 单份练习文件继续保留既有文件名前导序号兼容规则；复制为普通 `input.json` 时改用脱敏内容指纹。数组则使用稳定位置 ID。两种包装方式的自动 ID 可能不同，但来源内容和编号一致。
- 多病例练习数组不接受单值 `--practice-case-id` 或 `--practice-record-cutoff` 覆盖，避免把同一值错误应用到全部病例；每例应提供自身记录时间。
- 当前正式协议没有定义任意额外元数据的通用透传容器；已明确实现并验证的是现有 `Case/Source` Schema 字段。
- 第 07 份病例需先由资料提供方修复 JSON 语法，之后才能完成医学数据逐项追溯验收。
