import argparse
import zipfile
from pathlib import Path


INCLUDE = ["main.py", "requires.txt", "requirements.txt", "README.md", "resource_manifest.json", ".env.example", "src", "prompts", "resources",
           "config/competition.example.json", "config/local.mlx.example.json", "tools/adapt_practice.py",
           "config/retrieval.local.example.json", "tools/eval_clinical_pipeline.py",
           "tools/check_input.py",
           "tools/inspect_clinical_graph.py", "tools/smoke_graph_retrieval.py",
           "docs/clinical_experiment.md",
           "docs/algorithm_optimization_v043.md", "docs/graph_rag_multi_agent_design.md",
           "docs/clinical_verifier_revision_optimization.md", "docs/d_agent_failure_repair.md",
           "docs/automatic_input_adapter.md", "docs/d_graph_agent_integration_validation.md",
           "docs/competition_runtime_readiness.md",
           "docs/competition_advanced_pipeline_acceptance.md",
           "docs/unified_entrypoint_full_pipeline_acceptance.md",
           "docs/dual_json_input_compatibility.md",
           "docs/clinical_writing_research_synthesis.md",
           "docs/prompt_v2_shared_memory_writing_knowledge.md",
           "docs/competition_retrieval_fallback.md",
           "docs/final_quality_gate_v1.md",
           "docs/final_quality_gate_v1_regression_fix.md",
           "docs/final_validation_recovery_v1.md"]


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
