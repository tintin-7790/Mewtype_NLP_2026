# Mewtype Final Quality Gate v1 — 第一次真实回归修复

日期：2026-11-25
范围：只修复 Claim 定位、Local Reviser 闭环、deterministic fallback 与 Final Validation；未修改五阶段 Prompt、Graph、Agent、Shared Memory、比赛 Profile 或正式输出 Schema，未调用付费 Qwen。

## 1. 真实运行链路

本轮只读检查了最新 `run_log.json`、`output.json` 和两组自动备份，没有覆盖历史结果。

最新真实运行的五个模型阶段均调用、JSON 解析和阶段 Schema 校验成功。Verifier 给出 5 个 HIGH issue：C010、C014、C016、C029 涉及运行时 ID，其中 C014 同时触发弱阴性过度解释。

Reviser 实际返回并应用了 C010、C014、C016、C029 四个 Claim Patch，但 Patch 后确定性复查仍在 C014 检出 HIGH。`run_local_revision` 因此抛出：

```text
LOCAL_REVISION_UNRESOLVED_RISK
remaining_claim_ids=["C014"]
revised_claim_ids=["C010","C014","C016","C029"]
```

所以 `revision_applied=False` 并不表示 Reviser 没返回 Patch，而是旧实现发现残余风险后抛错，没有把部分修订结果返回 Pipeline。Pipeline 随后在原 Writer 记录上执行 `deterministic_conservative_fallback`。

普通日志按设计没有保存完整模型响应；本次未显式启用虚构病例 raw-response debug，因此无法从现有文件逐字恢复本次 Writer 初稿和 Reviser replacement。可以可靠确认的证据是 issue 定位/哈希、Patch Claim 列表、Patch 后残余 C014、fallback 前后正文长度 306→194、相似度 0.56，以及最终损坏文本。历史备份还保留了同类长句和括号内鉴别结构，可用于离线回归。

## 2. 根因

旧 `claim_extractor` 在长句或含多个数值时，会继续按中文逗号拆 Claim。C010/C014/C016 因而不是完整医学句，而可能是括号内部或鉴别列表的半个分句。

旧 fallback 对每个 HIGH Claim 都调用 `_fallback_text`，将其替换为：

```text
该项陈述依据不足，需医生结合原始资料确认。
```

替换范围虽能按字符 span 命中，但语义范围不完整。半句被替换后，相邻的 `AKI叠加？原发肾小球病？）。` 等内容被遗留，形成残句与括号错配。review_items 中的目标也被替换成同一无对象模板。

模板替换移除了 F/PR/C ID 和原来的弱阴性措辞，旧 Final Quality Gate 因而不再检测到原 HIGH；fallback 又把“有 edit”当成“已解决”，最终错误记录 `unresolved_risk_count=0`。

## 3. 修复

### 3.1 Claim 范围

- 正文 Claim 只按完整句、分号或换行切分，不再按逗号拆分。
- `apply_revision` 再次校验 span、原文完全匹配、左右句界及原 Claim 括号平衡。
- replacement 与原文相同、括号不匹配、缺少完整句级范围或使用通用审稿模板时，明确拒绝 Patch。

### 3.2 Reviser 闭环

- Model Patch 通过结构检查但仍有 HIGH 时，不再抛错并丢弃全部已验证 Patch。
- 返回部分修订结果，同时在 `patch_acceptance.post_patch_high_risk_claim_ids` 和 `unresolved_claim_ids` 中保留残余 HIGH。
- Pipeline 将状态设为 `completed_revision_unresolved`；启用 `block_unresolved_high_risk` 的本地完整 Profile/高级比赛 Profile 会阻止其作为 success 输出。

### 3.3 Conservative fallback

- 删除了 `_fallback_text` 通用模板路径。
- 只有完整列表元素的技术 review item、完整 evidence 元素可整体移除。
- 只有少数能够保留具体临床对象的“未知状态被写成确定未执行”句式允许确定性改写，例如把“当前尚未开始抗感染治疗”改为“当前是否已开始抗感染治疗尚未明确，需医生确认”。
- 内部 ID、弱阴性、因果或诊断推断等正文问题若没有安全确定性修法，原文保持不变，并记录 `preserved_unsafe_claims` 与真实 unresolved；不再用模板掩盖风险。

### 3.4 Final Validation

在既有 Schema、来源、数值、否定及时序检查之外，新增：

- `PROSE_BRACKET_MISMATCH`：正文括号不匹配。
- `GENERIC_REVISION_TEMPLATE`：正文出现无具体对象的审稿模板。
- `NON_ACTIONABLE_REVIEW_ITEM`：review_items 缺少具体可操作临床对象。

这些是输出质量错误，不通过删除日志或修改状态字段规避；必须由安全 Claim Patch 产生完整临床文本，否则运行失败或保留 unresolved。

## 4. 离线回归

`tests/test_final_quality_gate.py` 现有 25 项，新增场景直接取自本次实际损坏输出和历史备份结构：

- 实际重复模板、`AKI叠加？原发肾小球病？）。` 的括号破损被 Final Validation 拒绝。
- 带 F002 的长鉴别句不会再被 fallback 局部模板替换。
- fallback 不向 review_items 插入通用审稿模板。
- 长句中的逗号、括号内鉴别方向保持为一个完整可编辑 Claim。
- 模型返回通用 replacement 会被拒绝。
- Reviser 删除 ID 但仍保留弱阴性过度解释时，Patch 可以保留，但 residual HIGH 和 unresolved 不得归零。

验收结果：

- 全量 unittest：`207/207` 通过。
- Python 3.11 `compileall`：通过。
- `git diff --check`：通过。
- 五阶段 Prompt 文件和 SHA-256：本次未修改，仍与 `prompt_sha256.json` 一致。
- 正式输出 JSON Schema：未修改。
- 付费 Qwen 调用：0。

## 5. 当前打包判断

代码层面的破坏性 fallback 路径已关闭，但正式打包仍应暂缓到一次受控真实 DEMO001 回归完成。验收应满足：

1. Reviser 成功时只修改完整句级 Claim。
2. Reviser 仍有 HIGH 时，日志和最终状态真实保留 unresolved。
3. diagnostic_discussion 无残句、括号错误或通用模板。
4. review_items 均有具体临床核实对象。
5. 若 Writer 仍泄漏内部 ID 且无法安全修复，应明确失败，而不是生成损坏的 success 病历。

赛方 Qwen 的 `json_schema` 支持、调用预算以及正式 BGE-M3 HTTP 契约仍是原有外部待确认项。本轮没有新增其他算法风险或模型调用。

## 6. 本轮修改文件

- `src/claim_extractor.py`
- `src/claim_verifier.py`
- `src/local_reviser.py`
- `src/validation.py`
- `tests/test_final_quality_gate.py`
- `docs/final_quality_gate_v1.md`
- `docs/final_quality_gate_v1_regression_fix.md`
- `tools/build_package.py`（仅登记本报告）

没有修改 Prompt、历史输入输出、Graph/Agent/Memory、Profile 或比赛协议；没有 Commit/Push。
