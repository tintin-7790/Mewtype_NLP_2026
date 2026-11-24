"""Prompt profile loading with at most one synthetic few-shot per stage."""

from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROFILES = {"legacy", "contrastive", "contrastive_fewshot"}


def load_clinical_prompt(stage: str, profile: str, case_text: str) -> tuple[str, str | None]:
    if profile not in PROFILES:
        raise ValueError(f"unknown prompt profile: {profile}")
    if stage not in ("planner", "writer"):
        raise ValueError(f"unsupported prompt stage: {stage}")
    if profile == "legacy":
        path = BASE_DIR / "prompts" / "archive" / "clinical-v1" / f"{stage}_system.txt"
        return path.read_text(encoding="utf-8"), None
    prompt = (BASE_DIR / "prompts" / f"{stage}_system.txt").read_text(encoding="utf-8")
    if profile != "contrastive_fewshot":
        return prompt, None
    library_path = BASE_DIR / "prompts" / "fewshot" / f"{stage}_examples.json"
    library = json.loads(library_path.read_text(encoding="utf-8"))
    selected = max(library, key=lambda item: (
        sum(term in case_text for term in item["selector_terms"]), -item["order"]
    ))
    return prompt + "\n\n本例仅参考以下一个合成教学示例的组织方式，不得复制其中事实：\n" + selected["text"], selected["id"]
