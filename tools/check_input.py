"""Read-only input protocol and internal Case validation utility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.errors import CaseError, TaskError
from src.input_adapter import SUPPORTED_INPUT_FORMATS, load_input_cases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="只读检查 Mewtype 输入格式，不调用模型")
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-format", choices=SUPPORTED_INPUT_FORMATS, default="auto")
    args = parser.parse_args(argv)
    try:
        loaded = load_input_cases(Path(args.input), args.input_format)
        cases = []
        valid = True
        for item in loaded.cases:
            if isinstance(item, CaseError):
                valid = False
                cases.append({"case_id": item.case_id or "", "schema_valid": False,
                              "error_code": item.code})
            else:
                cases.append({"case_id": item.case_id, "source_count": len(item.sources),
                              "record_cutoff": item.record_cutoff, "schema_valid": True})
        print(json.dumps({"valid": valid, "detected_format": loaded.detected_format,
                          "case_count": len(loaded.cases), "cases": cases},
                         ensure_ascii=False, indent=2))
        return 0 if valid else 2
    except TaskError as exc:
        print(json.dumps({"valid": False, "error_code": exc.code,
                          "message": exc.message}, ensure_ascii=False), file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
