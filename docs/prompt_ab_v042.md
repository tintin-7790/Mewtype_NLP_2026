# Clinical Prompt v0.4 → v0.4.2 A/B 记录

v0.4 原始四份提示词保存在 `prompts/archive/clinical-v1/`。当前运行使用 `prompts/` 下的 v0.4.2 提示词，日志版本为 `2026-11-23.clinical-v3-neurosymbolic`。

| 阶段 | v0.4 字符数 | v0.4.2 字符数 | 增量 |
|---|---:|---:|---:|
| Planner | 936 | 1491 | +555 |
| Writer | 817 | 1329 | +512 |
| Verifier | 915 | 2116 | +1201 |
| Reviser | 662 | 825 | +163 |
| 合计 | 3330 | 5761 | +2431 |

字符数不是 API Token。真实 A/B 必须固定病例、Qwen 模型、参数、Schema 和检索配置，并分别读取 `run_log.json.stage_calls`。正常完整流程调用3次；仅出现 HIGH 且模型修订可用时调用第4次。
