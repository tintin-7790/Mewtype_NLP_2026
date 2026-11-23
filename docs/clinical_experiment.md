# Clinical Pipeline A–E 消融实验

实验必须固定病例版本、训练/开发/保留测试划分、Qwen 实际模型、生成参数、提示词版本和人工评审标准。相同模板衍生病例放在同一分区。Mock 只验证程序流程，不计入真实模型效果。

## 组别

| 组别 | 流程 | Qwen 调用预算 |
|---|---|---:|
| A | 原 Baseline | 1 |
| B | Planner + Writer | 2 |
| C | B + Dense Retrieval | 2 |
| D | C + Claim Verification | 3 |
| E | D + 必要时 Localized Revision | 4（正常通常3） |

D 检出 high 问题时不调用模型修订，而是对可精确定位的陈述作保守替换并披露；无法形成合法输出时才失败。medium 作为待核事项保留，low 不触发额外调用。E 仅对 high 问题最多进行一轮模型局部修订，修订失败时使用相同保守回退。每组分别记录 Qwen 与 Embedding 调用、Token、耗时、内部核验状态和未解决风险数。

## 评测 manifest

`tools/eval_clinical_pipeline.py` 接受：

```json
{
  "runs": [
    {
      "name": "A_baseline",
      "input": "data/fixed_cases.json",
      "output": "runs/a/output.json",
      "log": "runs/a/run_log.json",
      "adjudication": "annotations/a.json"
    }
  ]
}
```

所有相对路径按 manifest 所在目录解析。`adjudication` 可省略；省略后不计算需要人工真值的临床质量和检索指标。

人工标注格式：

```json
{
  "cases": {
    "CASE001": {
      "supported_fact_claims": 8,
      "total_fact_claims": 10,
      "negation_errors": 0,
      "temporal_status_errors": 1,
      "unsupported_claims": 1,
      "diagnostic_basis_covered": 3,
      "diagnostic_basis_total": 4,
      "plan_problem_links": 3,
      "plan_items_total": 4,
      "revision_attempts": 1,
      "revision_successes": 1,
      "unchanged_sections": 2,
      "expected_unchanged_sections": 2,
      "retrieval_ranks": [1, 3, null]
    }
  }
}
```

`retrieval_ranks` 来自人工标注正确证据片段在 Top-K 中的名次；`null` 表示未召回。不能使用 Writer 自己引用的来源或模型自评作为检索真值。

运行：

```bash
python3.11 tools/eval_clinical_pipeline.py \
  --manifest experiments/manifest.json \
  --output experiments/summary.json
```

报告每项指标的分子、分母和值。事实支持、诊断依据和计划对应需要医学或合格人工审核；规则只能作为初筛，不能替代医学评价。
