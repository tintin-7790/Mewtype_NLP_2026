# Clinical Pipeline A–E 消融实验

实验必须固定病例版本、训练/开发/保留测试划分、Qwen 实际模型、生成参数、提示词版本和人工评审标准。相同模板衍生病例放在同一分区。Mock 只验证程序流程，不计入真实模型效果。

## 组别

| 组别 | 流程 | Qwen 调用预算 |
|---|---|---:|
| A | 原 Baseline | 1 |
| B | 归档 Planner + Writer（`legacy`） | 2 |
| C | Contrastive Planner + Problem-Conditioned Writer | 2 |
| D | C + Claim Verification + 必要时 Localized Revision | 4（正常通常3） |
| E | D + 实际可用的 BGE-M3 | 与D相同Qwen预算，另计Embedding |

D/E 仅对 high 问题最多进行一轮模型局部修订，修订失败时使用保守回退。Verifier 自身 JSON/verdict 失败时保留合法 Writer 草稿，并在内部日志标记核验未完成。medium 作为待核事项保留，low 不触发额外调用。每组分别记录 Qwen 与 Embedding 调用、Token、耗时、内部核验状态和未解决风险数。`contrastive_fewshot` 可作为额外 C2 实验，每阶段最多注入一个合成示例，不与默认 C 混记。

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
      "limiting_evidence_accurate": 2,
      "limiting_evidence_total": 2,
      "critical_fact_errors": 0,
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

自动报告三段正文完整率、字符数、结构、合法来源、调用量、Token和耗时。事实支持、关键事实错误、诊断依据、反对/限制依据准确性和计划对应需要医学或合格人工审核；规则只能作为初筛，不能替代医学评价。
