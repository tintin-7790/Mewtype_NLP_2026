"""Case-scoped dense evidence retrieval with conservative compatibility hints."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

from .embedding_client import EmbeddingClient
from .facts import TIME_RE
from .protocol import Case, Source


@dataclass(frozen=True)
class RetrievalUnit:
    case_id: str
    source_id: str
    field_path: str
    text: str
    time: str | None
    assertion: str
    status: str
    ordinal: int


@dataclass(frozen=True)
class EvidenceCandidate:
    unit: RetrievalUnit
    similarity: float
    compatibility_warnings: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.compatibility_warnings

    def to_dict(self) -> dict:
        return {
            **asdict(self.unit),
            "similarity": round(self.similarity, 6),
            "compatible": self.compatible,
            "compatibility_warnings": list(self.compatibility_warnings),
        }


class DenseEvidenceRetriever:
    """Index and query only one case, preventing cross-patient retrieval."""

    def __init__(
        self,
        case: Case,
        client: EmbeddingClient,
        *,
        max_chunk_chars: int = 240,
        min_similarity: float = 0.0,
    ):
        if max_chunk_chars < 40:
            raise ValueError("max_chunk_chars must be at least 40")
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1")
        self.case = case
        self.client = client
        self.max_chunk_chars = max_chunk_chars
        self.min_similarity = min_similarity
        self.units = build_retrieval_units(case, max_chunk_chars=max_chunk_chars)
        self._source_vectors: list[list[float]] | None = None
        self._query_cache: dict[str, list[float]] = {}
        self._result_cache: dict[tuple[str, int], list[EvidenceCandidate]] = {}
        self.query_count = 0
        self._calls_at_start = client.total_calls
        self._texts_at_start = client.total_texts
        self._elapsed_at_start = client.elapsed_ms

    def retrieve(self, statement: str, top_k: int = 3) -> list[EvidenceCandidate]:
        if top_k <= 0 or not statement.strip() or not self.units:
            return []
        cache_key = (statement, top_k)
        if cache_key in self._result_cache:
            return list(self._result_cache[cache_key])
        self.query_count += 1
        self._ensure_index()
        query_vector = self._query_cache.get(statement)
        if query_vector is None:
            query_vector = self.client.embed([statement])[0]
            self._query_cache[statement] = query_vector
        query_meta = _semantic_metadata(statement)
        ranked = []
        for unit, vector in zip(self.units, self._source_vectors or []):
            score = cosine_similarity(query_vector, vector)
            if score < self.min_similarity:
                continue
            ranked.append(EvidenceCandidate(unit, score, _compatibility_warnings(query_meta, unit)))
        ranked.sort(key=lambda item: (-item.similarity, item.unit.ordinal))
        self._result_cache[cache_key] = ranked[:top_k]
        return list(self._result_cache[cache_key])

    def stats(self) -> dict:
        return {
            "provider": type(self.client).__name__,
            "embedding_calls": self.client.total_calls - self._calls_at_start,
            "embedded_texts": self.client.total_texts - self._texts_at_start,
            "embedding_elapsed_ms": round(self.client.elapsed_ms - self._elapsed_at_start, 2),
            "retrieval_queries": self.query_count,
            "indexed_units": len(self.units),
            "source_vector_cache": self._source_vectors is not None,
            "query_cache_entries": len(self._query_cache),
            "result_cache_entries": len(self._result_cache),
        }

    def _ensure_index(self) -> None:
        if self._source_vectors is None:
            self._source_vectors = self.client.embed([unit.text for unit in self.units]) if self.units else []


def build_retrieval_units(case: Case, *, max_chunk_chars: int = 240) -> list[RetrievalUnit]:
    units: list[RetrievalUnit] = []
    for source in case.sources:
        for text in _split_source(source, max_chunk_chars):
            metadata = _semantic_metadata(text)
            units.append(RetrievalUnit(
                case_id=case.case_id,
                source_id=source.source_id,
                field_path=source.field_path,
                text=text,
                time=metadata["time"],
                assertion=metadata["assertion"],
                status=metadata["status"],
                ordinal=len(units),
            ))
    return units


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _split_source(source: Source, max_chars: int) -> list[str]:
    statements = [part.strip() for part in re.split(r"(?<=[。！？!?；;])|[\r\n]+", source.content) if part.strip()]
    if not statements:
        statements = [source.content.strip()]
    chunks: list[str] = []
    for statement in statements:
        if len(statement) <= max_chars:
            chunks.append(statement)
            continue
        clauses = [part.strip() for part in re.split(r"(?<=[，,])", statement) if part.strip()]
        current = ""
        for clause in clauses:
            if current and len(current) + len(clause) > max_chars:
                chunks.extend(_fixed_chunks(current, max_chars))
                current = clause
            else:
                current += clause
        if current:
            chunks.extend(_fixed_chunks(current, max_chars))
    return chunks


def _fixed_chunks(text: str, max_chars: int) -> list[str]:
    return [text[start:start + max_chars] for start in range(0, len(text), max_chars) if text[start:start + max_chars].strip()]


def _semantic_metadata(text: str) -> dict:
    if any(word in text for word in ("未提供", "不详", "未知", "未查")):
        assertion = "not_provided"
    elif any(word in text for word in ("待排", "考虑", "可能", "？", "?", "尚不能")):
        assertion = "uncertain"
    elif any(word in text for word in ("否认", "未见", "无明显", "未闻及", "阴性")):
        assertion = "negative"
    else:
        assertion = "positive"
    if any(word in text for word in ("待完善", "拟行", "计划", "建议", "待复查")):
        status = "planned"
    elif any(word in text for word in ("正在", "目前使用", "长期服用", "继续")):
        status = "ongoing"
    elif any(word in text for word in ("已行", "曾行", "检查提示", "予以", "既往予", "曾予", "服用", "完成")):
        status = "completed"
    else:
        status = "unknown"
    time_match = TIME_RE.search(text)
    return {"assertion": assertion, "status": status, "time": time_match.group(0) if time_match else None}


def _compatibility_warnings(query: dict, unit: RetrievalUnit) -> tuple[str, ...]:
    warnings = []
    if {query["assertion"], unit.assertion} == {"positive", "negative"}:
        warnings.append("assertion_mismatch")
    known_statuses = {"planned", "ongoing", "completed"}
    if query["status"] in known_statuses and unit.status in known_statuses and query["status"] != unit.status:
        warnings.append("status_mismatch")
    if query["time"] and unit.time and query["time"] != unit.time:
        warnings.append("time_mismatch")
    return tuple(warnings)
