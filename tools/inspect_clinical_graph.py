#!/usr/bin/env python3
"""Build an inspectable clinical graph without calling Qwen or an embedding API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_graph import build_clinical_graph
from src.clinical_planner import number_facts, validate_clinical_plan
from src.facts import organize_facts
from src.input_adapter import load_input_cases
from src.io_utils import atomic_write_json
from src.protocol import CaseError


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="离线检查 Mewtype 病例内临床证据图")
    value.add_argument("--input", required=True, help="正式协议或练习嵌套 JSON")
    value.add_argument("--output", required=True, help="图审计 JSON 输出")
    value.add_argument("--input-format", choices=("auto", "competition", "practice"), default="auto")
    value.add_argument("--practice-case-id")
    value.add_argument("--practice-record-cutoff")
    value.add_argument("--plan", help="可选：单病例 clinical_plan JSON，加入问题/鉴别/计划层")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    plan = None
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    loaded = load_input_cases(Path(args.input), args.input_format,
                              practice_case_id=args.practice_case_id,
                              practice_record_cutoff=args.practice_record_cutoff)
    cases = loaded.cases
    output = []
    valid_cases = [case for case in cases if not isinstance(case, CaseError)]
    if plan is not None and len(valid_cases) != 1:
        raise SystemExit("--plan 当前仅支持恰好一个有效病例。")
    for case in cases:
        if isinstance(case, CaseError):
            output.append({"case_id": getattr(case, "case_id", ""), "status": "failed",
                           "error": {"code": case.code, "message": case.message}})
            continue
        facts = number_facts(organize_facts(case))
        if plan is not None:
            validate_clinical_plan(plan, {fact["fact_id"] for fact in facts})
        graph = build_clinical_graph(case, facts, plan)
        output.append({"case_id": case.case_id, "status": "success",
                       "record_cutoff": case.record_cutoff,
                       "input_adapter": loaded.audit,
                       "notice": "离线确定性图审计；未调用 Qwen/BGE，不代表医学判断。",
                       "facts": facts, "graph": graph.to_dict()})
    atomic_write_json(Path(args.output), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
