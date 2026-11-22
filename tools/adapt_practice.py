import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.errors import TaskError
from src.practice_adapter import adapt_file


def main():
    p = argparse.ArgumentParser(description="将中文嵌套练习病历转换为确定性的内部来源结构")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--case-id", required=True); p.add_argument("--record-cutoff", required=True)
    args = p.parse_args()
    try: adapt_file(Path(args.input), Path(args.output), args.case_id, args.record_cutoff); return 0
    except TaskError as exc: print(f"{exc.code}: {exc.message}"); return exc.exit_code


if __name__ == "__main__": raise SystemExit(main())
