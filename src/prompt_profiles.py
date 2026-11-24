"""Stage-aware clinical prompt profiles without changing stage schemas."""

from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROFILES = {"legacy", "contrastive", "contrastive_fewshot", "contrastive_v2"}
STAGES = {"planner", "counter_evidence", "writer", "verifier", "reviser"}
_STAGE_FILES = {
    "planner": "planner_system.txt",
    "counter_evidence": "counter_evidence_system.txt",
    "writer": "writer_system.txt",
    "verifier": "claim_verify_system.txt",
    "reviser": "revise_system.txt",
}


def resolve_clinical_prompt_path(stage: str, profile: str) -> Path:
    if profile not in PROFILES:
        raise ValueError(f"unknown prompt profile: {profile}")
    if stage not in STAGES:
        raise ValueError(f"unsupported prompt stage: {stage}")
    filename = _STAGE_FILES[stage]
    if profile == "contrastive_v2":
        return BASE_DIR / "prompts" / "contrastive_v2" / filename
    if profile == "legacy" and stage in ("planner", "writer"):
        return BASE_DIR / "prompts" / "archive" / "clinical-v1" / filename
    return BASE_DIR / "prompts" / filename


def load_clinical_prompt(stage: str, profile: str, case_text: str) -> tuple[str, str | None]:
    prompt = resolve_clinical_prompt_path(stage, profile).read_text(encoding="utf-8")
    if profile != "contrastive_fewshot" or stage not in ("planner", "writer"):
        return prompt, None
    library_path = BASE_DIR / "prompts" / "fewshot" / f"{stage}_examples.json"
    library = json.loads(library_path.read_text(encoding="utf-8"))
    selected = max(library, key=lambda item: (
        sum(term in case_text for term in item["selector_terms"]), -item["order"]
    ))
    return prompt + "\n\n本例仅参考以下一个合成教学示例的组织方式，不得复制其中事实：\n" + selected["text"], selected["id"]
