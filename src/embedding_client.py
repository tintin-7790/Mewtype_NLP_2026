"""Dense embedding adapters used by the optional evidence retriever.

The competition HTTP contract has not been published.  This module therefore
does not guess an endpoint, request body, authentication method, or retry
policy for the official service.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Protocol, Sequence

from .errors import TaskError


class EmbeddingClient(Protocol):
    total_calls: int
    total_texts: int
    elapsed_ms: float

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector for each input text, preserving order."""


class MockEmbeddingClient:
    """Deterministic offline client with optional exact vectors.

    Exact vectors make ranking expectations explicit in tests.  Unmapped text
    uses a stable hashed character-bigram vector; this is only a test fixture,
    not a semantic model and must never be reported as BGE-M3 output.
    """

    def __init__(self, vectors: dict[str, Sequence[float]] | None = None, dimension: int = 32):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._vectors = {text: _normalise_vector(vector) for text, vector in (vectors or {}).items()}
        configured_dimensions = {len(vector) for vector in self._vectors.values()}
        if len(configured_dimensions) > 1:
            raise ValueError("mock vectors must use one dimension")
        self.dimension = configured_dimensions.pop() if configured_dimensions else dimension
        self.total_calls = 0
        self.total_texts = 0
        self.elapsed_ms = 0.0
        self.batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        self.total_calls += 1
        self.total_texts += len(texts)
        self.batches.append(list(texts))
        result = []
        for text in texts:
            vector = self._vectors.get(text)
            result.append(list(vector) if vector is not None else _hashed_vector(text, self.dimension))
        self.elapsed_ms += (time.perf_counter() - started) * 1000
        return result


class LocalBgeEmbeddingClient:
    """Optional local FlagEmbedding adapter; never a formal runtime dependency.

    By default ``model`` must be an existing local path, preventing an implicit
    network download.  A developer may explicitly set ``allow_download=True``
    outside the competition environment to use a Hugging Face model id.
    """

    def __init__(
        self,
        model: str,
        *,
        use_fp16: bool = False,
        batch_size: int = 12,
        max_length: int = 8192,
        allow_download: bool = False,
    ):
        if not model.strip():
            raise TaskError("EMBEDDING_CONFIG_INVALID", "local_bge 必须显式配置 model。")
        if not allow_download and not Path(model).expanduser().exists():
            raise TaskError(
                "LOCAL_EMBEDDING_MODEL_MISSING",
                "本地 BGE 模型路径不存在；程序不会自动下载模型。",
            )
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise TaskError(
                "LOCAL_EMBEDDING_DEPENDENCY_MISSING",
                "local_bge 需要开发环境单独安装 FlagEmbedding；它不是正式提交依赖。",
            ) from exc
        self._model = BGEM3FlagModel(model, use_fp16=use_fp16)
        self.batch_size = batch_size
        self.max_length = max_length
        self.total_calls = 0
        self.total_texts = 0
        self.elapsed_ms = 0.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        started = time.perf_counter()
        self.total_calls += 1
        self.total_texts += len(texts)
        try:
            encoded = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vectors = encoded["dense_vecs"]
            result = [[float(value) for value in vector] for vector in vectors]
        except Exception as exc:
            raise TaskError("LOCAL_EMBEDDING_ERROR", f"本地 BGE 编码失败：{type(exc).__name__}") from exc
        finally:
            self.elapsed_ms += (time.perf_counter() - started) * 1000
        _validate_batch(texts, result, expected_dimension=1024)
        return result


class OfficialEmbeddingClient:
    """Deliberately unimplemented adapter boundary for the official intranet API."""

    def __init__(self):
        self.total_calls = 0
        self.total_texts = 0
        self.elapsed_ms = 0.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise TaskError(
            "OFFICIAL_EMBEDDING_PROTOCOL_PENDING",
            "赛方 BGE-M3 HTTP 请求格式、认证、超时、重试和配额尚未公布，未发起请求。",
        )


def load_embedding_client(path: Path, environment: str) -> tuple[EmbeddingClient, dict]:
    """Load the explicit optional retrieval configuration.

    The file is a development configuration, not part of the competition input
    protocol.  Competition mode permits only the official placeholder.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskError("EMBEDDING_CONFIG_INVALID", f"无法读取检索配置：{exc}") from exc
    if not isinstance(raw, dict):
        raise TaskError("EMBEDDING_CONFIG_INVALID", "检索配置必须是 JSON 对象。")
    provider = raw.get("provider")
    if environment == "competition" and provider != "official":
        raise TaskError("EMBEDDING_MODE_FORBIDDEN", "competition 模式只能使用赛方 official Embedding 适配器。")
    if provider == "mock":
        vectors = raw.get("vectors", {})
        if not isinstance(vectors, dict):
            raise TaskError("EMBEDDING_CONFIG_INVALID", "mock vectors 必须是对象。")
        client: EmbeddingClient = MockEmbeddingClient(vectors, int(raw.get("dimension", 32)))
        model_label = "deterministic-mock"
    elif provider == "local_bge":
        client = LocalBgeEmbeddingClient(
            str(raw.get("model", "")),
            use_fp16=bool(raw.get("use_fp16", False)),
            batch_size=int(raw.get("batch_size", 12)),
            max_length=int(raw.get("max_length", 8192)),
            allow_download=bool(raw.get("allow_download", False)),
        )
        model_label = str(raw.get("model", ""))
    elif provider == "official":
        allowed_official = {"provider", "top_k", "min_similarity", "max_chunk_chars"}
        unpublished = sorted(set(raw) - allowed_official)
        if unpublished:
            raise TaskError(
                "OFFICIAL_EMBEDDING_CONFIG_UNPUBLISHED",
                "赛方 Embedding 协议未公布，当前不能接受推测字段：" + ", ".join(unpublished),
            )
        client = OfficialEmbeddingClient()
        model_label = None
    else:
        raise TaskError("EMBEDDING_CONFIG_INVALID", "provider 必须为 mock/local_bge/official。")
    options = {
        "provider": provider,
        "model": model_label,
        "top_k": int(raw.get("top_k", 3)),
        "min_similarity": float(raw.get("min_similarity", 0.0)),
        "max_chunk_chars": int(raw.get("max_chunk_chars", 240)),
        "batch_size": int(raw.get("batch_size", 12)) if provider == "local_bge" else None,
        "max_length": int(raw.get("max_length", 8192)) if provider == "local_bge" else None,
        "use_fp16": bool(raw.get("use_fp16", False)) if provider == "local_bge" else None,
    }
    if options["top_k"] <= 0:
        raise TaskError("EMBEDDING_CONFIG_INVALID", "top_k 必须大于 0。")
    if not -1.0 <= options["min_similarity"] <= 1.0:
        raise TaskError("EMBEDDING_CONFIG_INVALID", "min_similarity 必须在 -1 到 1 之间。")
    if options["max_chunk_chars"] < 40:
        raise TaskError("EMBEDDING_CONFIG_INVALID", "max_chunk_chars 不得小于 40。")
    return client, options


def _validate_batch(texts: list[str], vectors: list[list[float]], expected_dimension: int | None = None) -> None:
    if len(vectors) != len(texts):
        raise TaskError("EMBEDDING_RESPONSE_INVALID", "向量数量与输入文本数量不一致。")
    dimensions = {len(vector) for vector in vectors}
    if texts and (not dimensions or 0 in dimensions or len(dimensions) != 1):
        raise TaskError("EMBEDDING_RESPONSE_INVALID", "向量维度为空或不一致。")
    if expected_dimension is not None and dimensions and dimensions != {expected_dimension}:
        raise TaskError("EMBEDDING_RESPONSE_INVALID", f"向量维度必须为 {expected_dimension}。")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise TaskError("EMBEDDING_RESPONSE_INVALID", "向量包含非有限数值。")


def _normalise_vector(vector: Sequence[float]) -> list[float]:
    result = [float(value) for value in vector]
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("mock vector must contain finite values")
    return result


def _hashed_vector(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    compact = "".join(text.split())
    features = [compact[index:index + 2] for index in range(max(1, len(compact) - 1))]
    if not compact:
        return vector
    for feature in features or [compact]:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        vector[index] += -1.0 if digest[4] & 1 else 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector
