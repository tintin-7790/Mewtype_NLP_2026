"""Build-time competition algorithm selection.

The organiser-facing .env remains limited to MODEL_NAME and ENDPOINT.  Change
ACTIVE_COMPETITION_PROFILE only when the official call/embedding budgets have
been confirmed, then rebuild and rerun the isolated acceptance suite.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import TaskError


@dataclass(frozen=True)
class CompetitionProfile:
    name: str
    pipeline: str
    qwen_calls_reserved: int
    environment: str = "competition"
    clinical_verification: bool = False
    clinical_revision: bool = False
    clinical_agents: str = "off"
    graph_retrieval: str = "off"
    retrieval_strategy: str = "auto"
    requires_official_embedding: bool = False
    requires_mock_embedding: bool = False
    clinical_prompt_profile: str = "contrastive"
    clinical_writing_guidance: str = "off"
    shared_clinical_memory: bool = False
    clinical_response_format: str | None = None
    block_unresolved_high_risk: bool = False
    final_validation_mode: str = "lite"
    fallback_profile: str = "baseline_safe"


PROFILES = {
    "baseline_safe": CompetitionProfile(
        name="baseline_safe", pipeline="baseline", qwen_calls_reserved=1,
    ),
    "clinical_d": CompetitionProfile(
        name="clinical_d", pipeline="clinical_pipeline", qwen_calls_reserved=4,
        clinical_verification=True, clinical_revision=True,
        block_unresolved_high_risk=True,
    ),
    "clinical_d_agent": CompetitionProfile(
        name="clinical_d_agent", pipeline="clinical_pipeline", qwen_calls_reserved=5,
        clinical_verification=True, clinical_revision=True, clinical_agents="always",
        block_unresolved_high_risk=True,
    ),
    "clinical_d_graph_agent": CompetitionProfile(
        name="clinical_d_graph_agent", pipeline="clinical_pipeline", qwen_calls_reserved=5,
        clinical_verification=True, clinical_revision=True, clinical_agents="always",
        graph_retrieval="ppr", retrieval_strategy="dense_graph",
        requires_official_embedding=True,
        clinical_prompt_profile="contrastive_v2",
        block_unresolved_high_risk=True,
    ),
    "clinical_d_lexical_graph_agent": CompetitionProfile(
        name="clinical_d_lexical_graph_agent", pipeline="clinical_pipeline",
        qwen_calls_reserved=5, clinical_verification=True, clinical_revision=True,
        clinical_agents="always", graph_retrieval="ppr",
        retrieval_strategy="lexical_graph", clinical_prompt_profile="contrastive_v2",
        block_unresolved_high_risk=True,
    ),
    "clinical_d_no_graph_agent": CompetitionProfile(
        name="clinical_d_no_graph_agent", pipeline="clinical_pipeline",
        qwen_calls_reserved=5, clinical_verification=True, clinical_revision=True,
        clinical_agents="always", graph_retrieval="off", retrieval_strategy="no_graph",
        clinical_prompt_profile="contrastive_v2", block_unresolved_high_risk=True,
    ),
    "competition_full_v2": CompetitionProfile(
        name="competition_full_v2",
        pipeline="clinical_pipeline",
        qwen_calls_reserved=5,
        clinical_verification=True,
        clinical_revision=True,
        clinical_agents="always",
        graph_retrieval="ppr",
        retrieval_strategy="lexical_graph",
        clinical_prompt_profile="contrastive_v2",
        clinical_writing_guidance="rules_examples",
        shared_clinical_memory=True,
        clinical_response_format="text",
        block_unresolved_high_risk=True,
        final_validation_mode="lite",
        fallback_profile="baseline_safe",
    ),
}


LOCAL_DEVELOPMENT_PROFILE = CompetitionProfile(
    name="development_clinical_d_graph_agent",
    environment="dev_api",
    pipeline="clinical_pipeline",
    qwen_calls_reserved=5,
    clinical_verification=True,
    clinical_revision=True,
    clinical_agents="always",
    graph_retrieval="ppr",
    retrieval_strategy="dense_graph",
    requires_mock_embedding=True,
    clinical_prompt_profile="contrastive_v2",
    clinical_writing_guidance="rules_examples",
    shared_clinical_memory=True,
    clinical_response_format="json_schema",
    block_unresolved_high_risk=True,
    final_validation_mode="strict",
)


# Submission default selected after the isolated Python 3.11, zero-argument,
# no-authentication Chat Completions acceptance suite.  It deliberately uses
# lexical seeds + PPR and therefore does not depend on the unpublished BGE API.
# This is a build-time choice and is intentionally not read from .env.
ACTIVE_COMPETITION_PROFILE = "competition_full_v2"


def get_active_competition_profile() -> CompetitionProfile:
    profile = PROFILES.get(ACTIVE_COMPETITION_PROFILE)
    if profile is None:
        raise TaskError("COMPETITION_PROFILE_INVALID",
                        f"未知正式算法 Profile：{ACTIVE_COMPETITION_PROFILE}")
    return profile


def profile_cli_arguments(profile: CompetitionProfile) -> list[str]:
    args = ["--mode", profile.environment, "--pipeline", profile.pipeline]
    if profile.pipeline == "clinical_pipeline":
        args.extend([
            "--clinical-verification", "on" if profile.clinical_verification else "off",
            "--clinical-revision", "on" if profile.clinical_revision else "off",
            "--clinical-agents", profile.clinical_agents,
            "--graph-retrieval", profile.graph_retrieval,
            "--retrieval-strategy", profile.retrieval_strategy,
            "--clinical-prompt-profile", profile.clinical_prompt_profile,
            "--clinical-writing-guidance", profile.clinical_writing_guidance,
            "--shared-clinical-memory", "on" if profile.shared_clinical_memory else "off",
        ])
        if profile.clinical_response_format:
            args.extend(["--clinical-response-format", profile.clinical_response_format])
    return args
