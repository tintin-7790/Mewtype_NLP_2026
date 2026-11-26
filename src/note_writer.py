"""Evidence-grounded note writer for the clinical pipeline."""

from __future__ import annotations

import json
import time

from .agent_orchestrator import attach_counter_review
from .model_client import Budget, MEDICAL_SCHEMA, parse_model_json
from .problem_context import build_problem_evidence_context
from .prompt_profiles import load_clinical_prompt
from .protocol import Case
from .writing_style_retriever import infer_tags, render_writer_guidance, select_exemplar_lessons


def run_note_writer(case: Case, facts: list[dict], clinical_plan: dict, evidence_bundles: list[dict],
                    client, response_format: str, budget: Budget, evidence_limit: int,
                    prompt_profile: str = "contrastive", prompt_audit: dict | None = None,
                    counter_review: dict | None = None,
                    writing_guidance: str = "off",
                    shared_memory_context: dict | None = None) -> dict:
    preparation_started = time.perf_counter()
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
        attach_counter_review(problem_context, counter_review)
        payload = {**shared, "problem_evidence_context": problem_context}
        if prompt_profile != "contrastive_v2":
            # Preserve the established v1 contrastive/few-shot input contract.
            # v2 uses the resolved problem context plus Writer-specific memory
            # and intentionally avoids duplicating the full plan.
            payload["clinical_plan"] = clinical_plan
    case_text = "\n".join(source.content for source in case.sources)
    selected_tags: set[str] = set()
    exemplar_ids: list[str] = []
    guidance_chars = 0
    if writing_guidance != "off":
        problem_text = " ".join(str(problem.get("name", ""))
                                for problem in clinical_plan.get("problems", []))
        selected_tags = infer_tags(problem_text) or infer_tags(case_text)
        include_examples = writing_guidance == "rules_examples"
        guidance = render_writer_guidance(
            case_text=case_text,
            tags=selected_tags or None,
            max_chars=2200,
            include_exemplar_lessons=include_examples,
        )
        payload["writing_style_context"] = guidance
        guidance_chars = len(guidance)
        if include_examples:
            exemplar_ids = [item["example_id"] for item in
                            select_exemplar_lessons(tags=selected_tags, limit=1)]
    if shared_memory_context is not None:
        payload["shared_decision_cards"] = shared_memory_context
    system, example_id = load_clinical_prompt("writer", prompt_profile, case_text)
    if prompt_audit is not None:
        decision_cards = (shared_memory_context.get("decision_cards", [])
                          if isinstance(shared_memory_context, dict) else [])
        clinical_focus = (shared_memory_context.get("case_clinical_focus", {})
                          if isinstance(shared_memory_context, dict) else {})
        prompt_audit["writer"] = {"profile": prompt_profile, "fewshot_example_id": example_id,
                                  "system_prompt_chars": len(system),
                                  "payload_chars": len(json.dumps(payload, ensure_ascii=False)),
                                  "writing_guidance": writing_guidance,
                                  "writing_guidance_chars": guidance_chars,
                                  "writing_guidance_tags": sorted(selected_tags),
                                  "writing_exemplar_ids": exemplar_ids,
                                  "shared_memory_injected": shared_memory_context is not None,
                                  "problem_context_count": len(
                                      payload.get("problem_evidence_context", {}).get("problems", [])
                                  ),
                                  "shared_memory_card_count": len(decision_cards),
                                  "shared_memory_problem_ids": [
                                      card.get("problem_id") for card in decision_cards
                                      if isinstance(card, dict)
                                  ],
                                  "clinical_focus_counts": {
                                      key: len(value) for key, value in clinical_focus.items()
                                      if isinstance(value, list)
                                  },
                                  "clinical_focus_truncated_reference_counts":
                                      clinical_focus.get("truncated_reference_counts", {}),
                                  "preparation_elapsed_ms": round(
                                      (time.perf_counter() - preparation_started) * 1000, 3)}
    messages = _messages(system, payload)
    raw = client.call(messages, response_format, budget, stage="writer", schema=MEDICAL_SCHEMA)
    return parse_model_json(raw, stage="writer", client=client)


def _messages(system: str, payload: dict) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
