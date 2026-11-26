# 北京语言大学 Mewtype — Final Quality Gate v1

> 2026-11-25 补充：第一次真实 Qwen 回归暴露了确定性 fallback 对逗号级 Claim 的破坏性替换。本报告关于首次修复的记录予以保留；后续根因、修复与最新验收以 `final_quality_gate_v1_regression_fix.md` 为准。

日期：2026-11-24
范围：正式打包前的定向质量修复；未调用付费 Qwen，未修改正式输入/输出协议，未切换正式默认 Profile。

## 1. 真实 DEMO001 缺陷与根因

本轮只读审计了根目录现有 `input.json`、`output.json` 和 `run_log.json`，没有覆盖历史文件。

### 1.1 已确认缺陷

- 最终 `diagnostic_discussion` 含 `F002`，属于运行时 Fact ID 泄漏。
- “现有记录未见明确黑便、肉眼血尿”被增强为“明确否定……并削弱隐匿性失血可能性”，改变了弱阴性证据的强度。
- `review_items` 含 “F002错误引用来源编号”等内部纠错说明，不是临床待核事项。
- 历史日志记录 Verifier 共 5 个 issue（HIGH 4、MEDIUM 1），Reviser 仅修改 C014、C015、C016、C026；`diagnostic_discussion` 被列为 preserved，因此泄漏所在的 C009 未被修改，但 `unresolved_risk_count` 仍为 0。
- 历史普通日志没有保存 issue 的完整患者文本，这是正确的隐私边界；但也只保留了计数，无法从日志逐字还原 5 条模型 issue。新日志改为保存 code、severity、section、source_ids 与 statement SHA-256，不保存完整陈述。

### 1.2 根因链

1. Planner、Agent 和 Shared Memory 合法使用 Fact/Problem ID 进行阶段协调。
2. Writer Prompt 未明确禁止将这些内部 ID 写入面向医生的正文，Writer 因而复制了 `F002`。
3. 原确定性核验器不知道本次运行实际生成的 ID 集合，只能依赖模型核验，未形成稳定的硬边界。
4. Local Reviser 将“存在 edit”近似当作“问题已解决”，未拒绝原样 replacement，也未使用运行时 ID 集合复查修订结果。
5. Pipeline 把 MEDIUM issue 的技术性说明直接拼入 `review_items`，并曾把内部修订提示再次转换为医生待核项。
6. 最后只做 Schema/来源格式校验，没有把内部审计泄漏作为正式输出错误。

离线使用新规则重扫历史 `output.json`，明确检出：C009 的 `INTERNAL_AUDIT_LEAK` 与 `UNSUPPORTED_NEGATIVE`，以及 C027 的 `INTERNAL_AUDIT_LEAK`、`TECHNICAL_REVIEW_ITEM`。这证明修复针对的是通用边界，不是硬编码一份正确病历。

## 2. 修改内容

### 2.1 Writer 与 Prompt v2

- Writer 明确禁止在正文、evidence.statement 和 review_items 中输出 Fact/Problem/Claim/Counter Issue ID；正式 Source ID 只允许出现在 `evidence.source_ids`。
- 明确“未见明确/无明显”属于有限弱阴性，不能扩写成全面否定或据此排除隐匿性疾病。
- 明确单次肾功能异常不能直接确认 CKD/AKI，贫血和肌酐异常并存不能直接证明肾性贫血的因果关系。
- 诊疗计划改为问题导向的四级优先顺序，避免为增加篇幅罗列常规检查。
- 非 legacy Writer 不再同时发送完整 `clinical_plan` 和已经包含等价规划信息的 `problem_evidence_context`，移除一份明显重复上下文；原始 sources 仍完整保留。

### 2.2 确定性最终质量边界

- 依据本次运行实际生成的 Fact、Problem、Claim 和 Counter Issue ID 集合检测泄漏，不用宽泛正则猜测医学文本。
- 新增 `INTERNAL_AUDIT_LEAK` 与 `TECHNICAL_REVIEW_ITEM` 两个程序内部确定性 code；二者固定为 HIGH，但不扩展模型 Verifier 枚举，因此各阶段 JSON Schema 保持不变。
- 对弱阴性来源与强排除/“削弱可能性”表达做窄范围检查；“不能排除/无法排除”不会被误报。
- 技术性待核项不再追加到 `review_items`，只在内部日志保留哈希审计。
- 最终正文仍存在内部 ID 或技术待核项时，Final Validation 返回错误，不能静默标记 success。

### 2.3 Claim Patch 闭环

- replacement 与原 Claim 相同会返回 `LOCAL_REVISION_NO_EFFECT`。
- Patch 后重新执行结构、来源、数值/单位、否定/时序/诊断确定性及运行时 ID 检查。
- 日志新增 `patch_acceptance`，记录目标 Claim、实际修改 Claim、是否全部变化及 Patch 后 HIGH 数量。
- 技术 review item 的确定性保守回退是精确 remove，不把技术说明改写成临床待核项。
- 最终重新计算 HIGH issue，并据此更新 `unresolved_risk_count`；不再只信任 Reviser 返回了多少 edit。

### 2.4 安全日志

新增 `verification_issue_audit` 和 `final_quality_gate`：只记录 Claim 定位、section、code、severity、合法 source_ids、statement SHA-256、被抑制技术项计数及最终未解决风险，不记录完整病例或完整模型响应。

## 3. 最终质量检查顺序

1. Writer JSON Schema 与 source_id 格式检查。
2. evidence 数值/单位的确定性支持检查与受限安全修复。
3. 用本次运行 ID 集合执行内部标识、技术 review item、弱阴性、数值、否定、时序、计划状态、诊断确定性检查。
4. Verifier 基于原始 Sources 做模型核验；Shared Memory、Graph、Agent 和写作范例不作为患者事实。
5. HIGH issue 进入一次局部 Reviser；无效或未覆盖 Patch 不算解决。
6. Patch 后重新执行结构校验和全部确定性高风险检查。
7. MEDIUM issue 只有具备临床意义时才可进入 review_items；技术说明仅写内部审计。
8. evidence policy 再检查。
9. Final Quality Gate 重新提取 Claims、重算 HIGH 与内部泄漏；边界错误阻断 success。
10. 最终标准输出 Schema 校验与原子写入。

未新增 Qwen 调用；正常完整链仍最多为 Planner、Agent、Writer、Verifier、Reviser 五次，Reviser 仍最多一次。

## 4. Prompt 完整性

旧 ZIP 提供的哈希清单曾与 ZIP 成员不一致，该异常保留在 `integrity_note`。本轮修改并重新计算集成文件：

| Prompt | Final Quality Gate 前 | 当前 SHA-256 | 原因 |
|---|---|---|---|
| planner | `ff7270e5…` | `ff7270e571a37421d2895a6b897671e929cc94c9c2049bbd644e9fccb70573a0` | 未修改 |
| counter_evidence | `f5c3a5c6…` | `f5c3a5c6594737c86ebbe06b20dba513c49d8015584428632ea93a63deaa7627` | 未修改 |
| writer | `1a134a9d…` | `0a6fb752e8513b74362e1f6ebbc5706c8b95058f82a977106a01950f8b2afab1` | ID、弱阴性、肾功能/贫血与计划约束 |
| verifier | `e632f724…` | `a7d04f933ffbda9c0e1826740da4e905b09da8aacf69e775bd032a20618fb6d4` | 明确程序审计边界和弱阴性规则，模型枚举不变 |
| reviser | `6a229932…` | `5a690fd634715b93aa5cd352cfd830bf40edb201fffe5ce3208c8485b2c48f7e` | Patch 实效和技术项移除规则 |

自动测试逐文件核对实际加载路径、文件内容和 `prompt_sha256.json`。打包工具递归包含 `prompts/` 与 `resources/`，五阶段 Prompt、哈希清单、写作资源均在白名单内。

## 5. Shared Memory、写作知识与 Graph/BGE 审计

- Shared Clinical Memory 仍为单病例、进程内、非持久化；Fact ID 只映射当前 Case 的 Source，Verifier/Reviser 不读取 Memory。
- Writer 写作知识以 `WRITING_STYLE_REFERENCE (NOT_PATIENT_EVIDENCE)` 标记，仅提供组织和措辞；合成示例不分配患者 source_id。
- 模块关闭时原 Clinical Pipeline 仍由现有回归测试覆盖。
- Dense 接口仍是 `embed(texts: list[str]) -> list[list[float]]`，病例内批量建立 source 向量并缓存查询；余弦相似度自行按范数计算，PPR 只把检索结果当初始化种子，Source/Fact 映射不跨病例。
- Local BGE 路径验证 1024 维；Mock 只用于开发控制流。OfficialEmbeddingClient 保持明确未实现，不猜测 Endpoint、认证、请求/响应字段、批上限、超时、重试或配额。
- 正式 BGE 仍待赛方确认：完整 Endpoint、是否认证、请求字段、批量上限、1024 维返回字段、最大文本长度、超时、重试、限流及错误码。未完成官方真实联调。

## 6. 成本与性能审计

历史 DEMO001 共 5 次调用、26,497 tokens：

| 阶段 | Prompt | Completion | Total | 耗时 ms | 输入字符 |
|---|---:|---:|---:|---:|---:|
| Planner | 1,602 | 953 | 2,555 | 21,409.05 | 3,725 |
| Counter-Evidence | 4,701 | 210 | 4,911 | 5,226.26 | 10,540 |
| Writer | 9,844 | 826 | 10,670 | 19,078.23 | 22,527 |
| Verifier | 5,546 | 692 | 6,238 | 15,855.00 | 14,120 |
| Reviser | 1,857 | 266 | 2,123 | 6,276.72 | 4,127 |

Writer 是最大输入阶段。审计确认它同时携带原始 Sources、完整 clinical_plan、已展开同等计划内容的 problem_evidence_context、写作规则和 Shared Memory。本轮只移除重复的完整 clinical_plan；不裁剪 Sources、不删除关键事实，也不增加调用。实际节省量需在下一次唯一真实验收中测量，不能用离线结果冒充真实 Token 改善。

## 7. 回归与静态验收

- 新增 `tests/test_final_quality_gate.py`：19 项，覆盖运行时 Fact/Problem/Claim ID、调试标记、技术 review item、弱阴性、检查状态、待排诊断、无效 Patch、Patch 后复查、技术项移除、写作知识边界、Memory 映射/隔离、Prompt 哈希、Profile 隔离、Schema 及最终阻断。
- 全量 unittest：`201/201` 通过。
- Python 3.11 `compileall`：通过。
- `input.json`、`output.json`、`run_log.json` 和资源清单 JSON 解析：通过。
- 写作资源 SHA-256：全部与 `resource_manifest.json` 一致。
- contrastive_v2 Prompt SHA-256：全部与实际文件一致。
- 既有测试继续覆盖正式 Sources、中文入院记录、两参数入口、competition 禁止百炼/Mock Embedding、Graph/Agent、Verifier/Reviser 和最终 Schema。

## 8. Profile 与提交风险

- 本地 `MEWTYPE_DEV_MODE=1`：`development_clinical_d_graph_agent`，contrastive_v2 + Clinical + PPR + Agent + rules_examples + Shared Memory + Verifier/Reviser + json_schema；开发 Embedding 为明确标记的 Mock。
- 正式 competition：仍固定 `baseline_safe`；不使用百炼、不使用 Mock Embedding，不受本地开发环境变量自动切换。
- 统一命令保持：`python3.11 main.py --input input.json --output output.json`。

正式提交前仍有三项阻塞性外部风险：赛方 Qwen 是否支持分阶段 json_schema、正式调用预算/超时/重试上限、BGE-M3 HTTP 契约与配额。另需进行一次受控真实 Qwen 验收，确认 Writer 不再输出内部 ID、Reviser 对新 HIGH code 的 Patch 质量及实际 Token 变化。本轮没有把 Mock 成功描述为官方联调成功。

## 9. 本轮修改文件

- `src/claim_verifier.py`
- `src/local_reviser.py`
- `src/model_client.py`
- `src/note_writer.py`
- `src/pipeline.py`
- `prompts/contrastive_v2/writer_system.txt`
- `prompts/contrastive_v2/claim_verify_system.txt`
- `prompts/contrastive_v2/revise_system.txt`
- `prompts/contrastive_v2/prompt_sha256.json`
- `tests/test_final_quality_gate.py`
- `docs/final_quality_gate_v1.md`
- `tools/build_package.py`（仅把本报告加入既有文档白名单）

没有修改历史输入、输出或日志，没有制作 ZIP，没有 Commit/Push。
