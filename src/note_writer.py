"""Evidence-grounded note writer for the clinical pipeline."""

from __future__ import annotations

import json

from .model_client import Budget, MEDICAL_SCHEMA, parse_model_json
from .problem_context import build_problem_evidence_context
from .prompt_profiles import load_clinical_prompt
from .protocol import Case


def run_note_writer(case: Case, facts: list[dict], clinical_plan: dict, evidence_bundles: list[dict],
                    client, response_format: str, budget: Budget, evidence_limit: int,
                    prompt_profile: str = "contrastive", prompt_audit: dict | None = None) -> dict:
    shared = {
        "record_cutoff": case.record_cutoff,
        "sources": [{"source_id": source.source_id, "field_path": source.field_path,
                     "content": source.content} for source in case.sources],
        "retrieval_notice": "候选仅用于定位原文，相似度不代表事实成立；必须核对原始 sources。",
        "evidence_limit": evidence_limit,
    }
    if prompt_profile == "legacy":
        # Preserve the archived Writer prompt's original input contract.
        payload = {**shared, "facts": facts, "clinical_plan": clinical_plan,
                   "evidence_bundles": evidence_bundles}
    else:
        problem_context = build_problem_evidence_context(facts, clinical_plan, evidence_bundles)
        payload = {**shared, "clinical_plan": clinical_plan,
                   "problem_evidence_context": problem_context}
    case_text = "\n".join(source.content for source in case.sources)
    system, example_id = load_clinical_prompt("writer", prompt_profile, case_text)
    if prompt_audit is not None:
        prompt_audit["writer"] = {"profile": prompt_profile, "fewshot_example_id": example_id,
                                  "system_prompt_chars": len(system)}
    messages = _messages(system, payload)
    raw = client.call(messages, response_format, budget, stage="writer", schema=MEDICAL_SCHEMA)
    return parse_model_json(raw, stage="writer", client=client)


def _messages(system: str, payload: dict) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
