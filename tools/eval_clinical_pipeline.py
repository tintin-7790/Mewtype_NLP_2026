#!/usr/bin/env python3
"""Aggregate A-E clinical-pipeline runs without using model self-ratings as truth.

Manifest paths are resolved relative to the manifest file.  Human adjudication
is optional, but clinical quality and retrieval metrics are omitted unless it
is supplied explicitly.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.io_utils import atomic_write_json  # noqa: E402
from src.protocol import Case, load_standard_cases  # noqa: E402
from src.validation import validate_record  # noqa: E402


MANUAL_COUNTS = (
    "supported_fact_claims", "total_fact_claims", "negation_errors", "temporal_status_errors",
    "unsupported_claims", "diagnostic_basis_covered", "diagnostic_basis_total",
    "limiting_evidence_accurate", "limiting_evidence_total", "critical_fact_errors",
    "plan_problem_links", "plan_items_total", "revision_attempts", "revision_successes",
    "unchanged_sections", "expected_unchanged_sections",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="汇总 clinical_pipeline A-E 消融实验")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("runs"), list):
        raise SystemExit("manifest 必须包含 runs 数组")
    results = [_evaluate_run(run, manifest_path.parent) for run in manifest["runs"]]
    payload = {"warning": "人工临床标注缺失的指标不会由模型自评或相似度替代。", "runs": results}
    if args.output:
        atomic_write_json(Path(args.output), payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _evaluate_run(spec: dict, base: Path) -> dict:
    name = spec.get("name")
    cases = load_standard_cases(_path(base, spec["input"]))
    if any(not isinstance(case, Case) for case in cases):
        raise SystemExit(f"{name}: 输入包含无效病例")
    outputs = json.loads(_path(base, spec["output"]).read_text(encoding="utf-8"))
    log = json.loads(_path(base, spec["log"]).read_text(encoding="utf-8"))
    case_map = {case.case_id: case for case in cases}
    structure_ok = 0; evidence_total = 0; legal_evidence = 0; complete_sections = 0; body_lengths = []
    for item in outputs:
        case = case_map.get(item.get("case_id"))
        if not case or item.get("status") != "success":
            continue
        record = item.get("medical_record")
        try:
            issues = validate_record(record, case)
        except Exception:
            issues = [{"level": "error"}]
        if not any(issue.get("level") == "error" for issue in issues):
            structure_ok += 1
        sections = [record.get(key) for key in ("case_features", "diagnostic_discussion", "plan")]
        if all(isinstance(value, str) and value.strip() for value in sections):
            complete_sections += 1
            body_lengths.append(sum(len(value) for value in sections))
        known = {source.source_id for source in case.sources}
        for evidence in record.get("evidence", []) if isinstance(record, dict) else []:
            evidence_total += 1
            ids = evidence.get("source_ids") if isinstance(evidence, dict) else None
            if isinstance(ids, list) and ids and all(source_id in known for source_id in ids):
                legal_evidence += 1
    result = {
        "name": name,
        "case_count": len(cases),
        "structure_compliance": _ratio(structure_ok, len(cases)),
        "three_section_completeness": _ratio(complete_sections, len(cases)),
        "body_length_chars": {
            "count": len(body_lengths),
            "total": sum(body_lengths),
            "average": round(sum(body_lengths) / len(body_lengths), 2) if body_lengths else None,
            "minimum": min(body_lengths) if body_lengths else None,
            "maximum": max(body_lengths) if body_lengths else None,
        },
        "legal_source_citation": _ratio(legal_evidence, evidence_total),
        "model_calls": int(log.get("total_calls", 0)),
        "token_usage": log.get("token_usage", {}),
        "elapsed_ms": log.get("elapsed_ms"),
        "manual_metrics": None,
    }
    if spec.get("adjudication"):
        adjudication = json.loads(_path(base, spec["adjudication"]).read_text(encoding="utf-8"))
        result["manual_metrics"] = _manual_metrics(adjudication)
    return result


def _manual_metrics(adjudication: dict) -> dict:
    totals = {key: 0 for key in MANUAL_COUNTS}; ranks = []
    for item in adjudication.get("cases", {}).values():
        for key in MANUAL_COUNTS:
            totals[key] += int(item.get(key, 0))
        ranks.extend(item.get("retrieval_ranks", []))
    return {
        "fact_support_rate": _ratio(totals["supported_fact_claims"], totals["total_fact_claims"]),
        "negation_errors": totals["negation_errors"],
        "temporal_status_errors": totals["temporal_status_errors"],
        "unsupported_claims": totals["unsupported_claims"],
        "diagnostic_basis_coverage": _ratio(totals["diagnostic_basis_covered"], totals["diagnostic_basis_total"]),
        "limiting_evidence_accuracy": _ratio(totals["limiting_evidence_accurate"],
                                              totals["limiting_evidence_total"]),
        "critical_fact_errors": totals["critical_fact_errors"],
        "plan_problem_alignment": _ratio(totals["plan_problem_links"], totals["plan_items_total"]),
        "revision_success_rate": _ratio(totals["revision_successes"], totals["revision_attempts"]),
        "unchanged_section_preservation": _ratio(totals["unchanged_sections"], totals["expected_unchanged_sections"]),
        "retrieval_recall_at_k": _ratio(sum(rank is not None for rank in ranks), len(ranks)),
        "retrieval_mrr": (round(sum(1 / int(rank) for rank in ranks if rank) / len(ranks), 6) if ranks else None),
        "raw_counts": totals,
    }


def _ratio(numerator: int, denominator: int):
    return {"numerator": numerator, "denominator": denominator,
            "value": round(numerator / denominator, 6) if denominator else None}


def _path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


if __name__ == "__main__":
    raise SystemExit(main())
