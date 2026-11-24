# Clinical Prompt v0.4 → v0.4.3 对照

本记录只描述提示词和工作流差异，不代表临床质量已经提升。归档版本位于
`prompts/archive/clinical-v1/`，默认运行档位为 `contrastive`。

| 阶段 | legacy字符数 | contrastive字符数 | 增量 |
|---|---:|---:|---:|
| Planner | 936 | 1729 | +793 |
| Writer | 817 | 1800 | +983 |

`contrastive_fewshot` 每阶段只选择一个合成示例：Planner 实际 system prompt
约1901–2476字符，Writer约1950–1984字符，具体示例 ID 和字符数写入运行日志。
字符数不是 API Token；Token 成本必须读取真实服务 usage。

主要变化：

- Planner 从问题罗列改为支持证据、明确限制、待区分资料、少量鉴别和 plan goal 的对比式组织。
- Writer 新增 Python 确定构建的 `problem_evidence_context`，不再只接收平铺 JSON。
- Evidence Salience Ranking 提升数值、异常、明确阴性、更正、时序、状态、暴露和治疗史优先级，降低无关人口学字段优先级。
- Dense 查询可补入当前问题引用的患者事实和待区分资料，但相似度仍不作为事实证明。
- Few-shot 仅为可选消融档位，不增加 Qwen 调用次数，也不默认宣称有效。
