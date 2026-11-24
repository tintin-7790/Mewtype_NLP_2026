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
    requires_official_embedding: bool = False
    requires_mock_embedding: bool = False
    clinical_prompt_profile: str = "contrastive"
    clinical_response_format: str | None = None
    block_unresolved_high_risk: bool = False
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
        graph_retrieval="ppr", requires_official_embedding=True,
        block_unresolved_high_risk=True,
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
    requires_mock_embedding=True,
    clinical_prompt_profile="contrastive",
    clinical_response_format="json_schema",
    block_unresolved_high_risk=True,
)


# Conservative submission default until the official Qwen and BGE-M3 budgets
# and HTTP contracts are known.  This is intentionally not read from .env.
ACTIVE_COMPETITION_PROFILE = "baseline_safe"


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
            "--clinical-prompt-profile", profile.clinical_prompt_profile,
        ])
        if profile.clinical_response_format:
            args.extend(["--clinical-response-format", profile.clinical_response_format])
    return args
