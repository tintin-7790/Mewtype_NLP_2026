"""Evidence-grounded note writer for the clinical pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from .model_client import Budget, MEDICAL_SCHEMA, parse_model_json
from .protocol import Case


BASE_DIR = Path(__file__).resolve().parent.parent


def run_note_writer(case: Case, facts: list[dict], clinical_plan: dict, evidence_bundles: list[dict],
                    client, response_format: str, budget: Budget, evidence_limit: int) -> dict:
    payload = {
        "record_cutoff": case.record_cutoff,
        "sources": [{"source_id": source.source_id, "field_path": source.field_path,
                     "content": source.content} for source in case.sources],
        "facts": facts,
        "clinical_plan": clinical_plan,
        "evidence_bundles": evidence_bundles,
        "retrieval_notice": "候选仅用于定位原文，相似度不代表事实成立；必须核对原始 sources。",
        "evidence_limit": evidence_limit,
    }
    messages = _messages("writer_system.txt", payload)
    raw = client.call(messages, response_format, budget, stage="writer", schema=MEDICAL_SCHEMA)
    return parse_model_json(raw, stage="writer", client=client)


def _messages(prompt_name: str, payload: dict) -> list[dict]:
    system = (BASE_DIR / "prompts" / prompt_name).read_text(encoding="utf-8")
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
