import argparse
import zipfile
from pathlib import Path


INCLUDE = ["main.py", "requirements.txt", "README.md", "resource_manifest.json", "src", "prompts",
           "config/competition.example.json", "config/local.mlx.example.json", "tools/adapt_practice.py",
           "config/retrieval.local.example.json", "tools/eval_clinical_pipeline.py",
           "docs/clinical_experiment.md"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--team", required=True); p.add_argument("--version", required=True)
    p.add_argument("--output"); args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    archive = Path(args.output) if args.output else root / f"{args.team}_{args.version}.zip"
    prefix = f"{args.team}_{args.version}"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in INCLUDE:
            path = root / name
            paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
            for item in paths:
                zf.write(item, Path(prefix) / item.relative_to(root))
    print(archive)
    return 0


if __name__ == "__main__": raise SystemExit(main())
