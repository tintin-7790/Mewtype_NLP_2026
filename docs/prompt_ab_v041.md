# Clinical Prompt v0.4 → v0.4.1 A/B 记录

旧版四份提示词冻结在 `prompts/archive/clinical-v1/`；下表是 v0.4.1 开发时的历史测量。当前 v0.4.2 运行提示词另见 `prompt_ab_v042.md`。

| 阶段 | v0.4 字符数 | v0.4.1 字符数 | 增量 |
|---|---:|---:|---:|
| Planner | 936 | 1387 | +451 |
| Writer | 817 | 1206 | +389 |
| Verifier | 915 | 2022 | +1107 |
| Reviser | 662 | 762 | +100 |
| 合计 | 3330 | 5377 | +2047 |

字符数只用于静态版本比较，不能等同 API Token。实际 prompt/completion/total tokens 必须读取相同病例、模型、参数和 Schema 下的 `run_log.json.stage_calls`；Reviser 只在 HIGH 问题出现时调用，因此单独报告条件调用的 Token，不能把未发生的第四次调用记为零成本质量提升。
