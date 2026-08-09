"""Deterministic rescoring baselines over an existing retrieval document union."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
import json
import os
from pathlib import Path
import re
from statistics import median
import tempfile
from typing import Any, Mapping, Protocol, TypeVar

from trec_rag.chunking import SemanticTextChunker, TextChunk
from trec_rag.document_store import DocumentStore
from trec_rag.mixedbread_passage_scorer import ScoredPassage


_TOP_PASSAGE_WEIGHTS = (0.55, 0.25, 0.13, 0.07)
_Key = TypeVar("_Key")


@dataclass(frozen=True)
class PassageScore:
    """One finite model score bound to a half-open source character span."""

    start_char: int
    end_char: int
    raw_score: float

    def __post_init__(self) -> None:
        if isinstance(self.start_char, bool) or not isinstance(self.start_char, int):
            raise TypeError("start_char must be an integer")
        if isinstance(self.end_char, bool) or not isinstance(self.end_char, int):
            raise TypeError("end_char must be an integer")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError("passage spans must be nonempty half-open spans")
        if isinstance(self.raw_score, bool) or not isinstance(self.raw_score, int | float):
            raise TypeError("raw_score must be a finite real number")
        if not math.isfinite(self.raw_score):
            raise ValueError("raw_score must be finite")


def _validate_percentile(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return result


def _validate_raw_score(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class DocumentScore:
    """Normalized semantic scores and the stable source-rank tie break for a document."""

    docid: str
    best_retrieval_rank: int
    narrative_raw_score: float
    subnarrative_raw_scores: Mapping[str, float]
    narrative_percentile: float
    subnarrative_percentiles: Mapping[str, float]
    subnarrative_source_ranks: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.docid, str) or not self.docid:
            raise ValueError("docid must be a nonempty string")
        if (
            isinstance(self.best_retrieval_rank, bool)
            or not isinstance(self.best_retrieval_rank, int)
            or self.best_retrieval_rank <= 0
        ):
            raise ValueError("best_retrieval_rank must be a positive integer")
        _validate_raw_score(self.narrative_raw_score, label="narrative raw score")
        _validate_percentile(self.narrative_percentile, label="narrative percentile")
        if not self.subnarrative_percentiles:
            raise ValueError("at least one subnarrative percentile is required")
        if set(self.subnarrative_raw_scores) != set(self.subnarrative_percentiles):
            raise ValueError("raw and percentile subnarrative sets must match")
        if not set(self.subnarrative_source_ranks).issubset(
            self.subnarrative_percentiles
        ):
            raise ValueError("subnarrative source ranks contain an unknown unit")
        if any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
            for rank in self.subnarrative_source_ranks.values()
        ):
            raise ValueError("subnarrative source ranks must be positive integers")
        for subnarrative_id, value in self.subnarrative_percentiles.items():
            if not isinstance(subnarrative_id, str) or not subnarrative_id:
                raise ValueError("subnarrative IDs must be nonempty strings")
            _validate_percentile(value, label="subnarrative percentile")
            _validate_raw_score(
                self.subnarrative_raw_scores[subnarrative_id],
                label="subnarrative raw score",
            )

    @property
    def strongest_percentile(self) -> float:
        return max(self.narrative_percentile, *self.subnarrative_percentiles.values())

    @property
    def facet_score(self) -> float:
        strongest = sorted(self.subnarrative_percentiles.values(), reverse=True)
        if len(strongest) == 1:
            return strongest[0]
        return 0.7 * strongest[0] + 0.3 * strongest[1]

    @property
    def combo_score(self) -> float:
        return 0.5 * self.narrative_percentile + 0.5 * self.facet_score


@dataclass(frozen=True)
class BreadthPassage:
    """A passage scored against its pooled subnarrative semantic unit."""

    subnarrative_id: str
    docid: str
    passage: PassageScore

    def __post_init__(self) -> None:
        if not isinstance(self.subnarrative_id, str) or not self.subnarrative_id:
            raise ValueError("subnarrative_id must be a nonempty string")
        if not isinstance(self.docid, str) or not self.docid:
            raise ValueError("docid must be a nonempty string")
        if not isinstance(self.passage, PassageScore):
            raise TypeError("passage must be a PassageScore")


@dataclass(frozen=True)
class RunRankings:
    """The common eligible set and three deterministic document orderings."""

    eligible_docids: tuple[str, ...]
    narrative: tuple[str, ...]
    combo: tuple[str, ...]
    breadth: tuple[str, ...]


@dataclass(frozen=True)
class CutoffUnitStat:
    unit_id: str
    median: float
    mad: float
    threshold: float
    comparison: str
    admitted_count: int
    admitted_docids_sha256: str


@dataclass(frozen=True)
class CutoffDecision:
    units: tuple[CutoffUnitStat, ...]
    eligible_docids: tuple[str, ...]
    pre_fallback_count: int
    fallback_used: bool


@dataclass(frozen=True)
class RankedTopic:
    topic_id: str
    document_scores: tuple[DocumentScore, ...]
    breadth_passages: tuple[BreadthPassage, ...]
    cutoff: CutoffDecision
    rankings: RunRankings


@dataclass(frozen=True)
class SemanticUnit:
    """One subnarrative and the retrieval strings pooled beneath it."""

    subnarrative_id: str
    text: str
    retrieval_query_texts: tuple[str, ...]


@dataclass(frozen=True)
class SourceDocument:
    """One complete-union document with authenticated content."""

    docid: str
    content_sha256: str
    best_retrieval_rank: int
    subnarrative_source_ranks: Mapping[str, int]
    text: str


@dataclass(frozen=True)
class TopicInput:
    """Strict local input for one existing retrieval topic."""

    topic_id: str
    narrative: str
    subnarratives: tuple[SemanticUnit, ...]
    documents: tuple[SourceDocument, ...]
    source_sha256s: Mapping[str, str]


@dataclass(frozen=True)
class MatrixDocument:
    docid: str
    content_sha256: str
    best_retrieval_rank: int
    subnarrative_source_ranks: Mapping[str, int]


@dataclass(frozen=True)
class MatrixUnit:
    unit_id: str
    kind: str
    text_sha256: str


@dataclass(frozen=True)
class MatrixPassage:
    unit_id: str
    docid: str
    start_char: int
    end_char: int
    raw_score: float


@dataclass(frozen=True)
class TopicMatrix:
    topic_id: str
    documents: tuple[MatrixDocument, ...]
    units: tuple[MatrixUnit, ...]
    passages: tuple[MatrixPassage, ...]
    source_sha256s: Mapping[str, str]
    scorer_identity: Mapping[str, object]
    chunker_identity: Mapping[str, object]
    cache_stats: Mapping[str, int]


class PassageScorer(Protocol):
    @property
    def identity(self) -> Mapping[str, object]: ...

    @property
    def stats(self) -> Mapping[str, int]: ...

    def rank(
        self, query_text: str, chunks: tuple[TextChunk, ...]
    ) -> tuple[ScoredPassage, ...]: ...


def _overlap_coefficient(first: PassageScore, second: PassageScore) -> float:
    intersection = max(
        0,
        min(first.end_char, second.end_char)
        - max(first.start_char, second.start_char),
    )
    shorter = min(
        first.end_char - first.start_char,
        second.end_char - second.start_char,
    )
    return intersection / shorter


def topic_sort_key(topic_id: str) -> tuple[str, int, str]:
    """Sort topic identifiers by a trailing integer when present."""

    if not isinstance(topic_id, str):
        raise TypeError("topic_id must be a string")
    match = re.fullmatch(r"(.*?)(\d+)", topic_id)
    if match is None:
        return (topic_id, -1, topic_id)
    return (match.group(1), int(match.group(2)), topic_id)


def _read_json_object(path: Path) -> tuple[dict[str, object], str]:
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"required source file is unreadable: {path}") from exc
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"required source file is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"required source file must contain a JSON object: {path}")
    return value, sha256(body).hexdigest()


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def load_topic_input(
    source_dir: Path,
    topic_id: str,
    document_store_root: Path,
) -> TopicInput:
    """Load one complete source topic without opening organizer judgments."""

    if not isinstance(topic_id, str) or not topic_id or Path(topic_id).name != topic_id:
        raise ValueError("topic_id must be a safe path component")
    topic_dir = Path(source_dir) / topic_id
    relative_paths = (
        "topic-job-receipt.json",
        "decomposition/result.json",
        "retrieval/audit.json",
        "scoring/selection.json",
    )
    loaded: dict[str, dict[str, object]] = {}
    source_sha256s: dict[str, str] = {}
    for relative in relative_paths:
        value, digest = _read_json_object(topic_dir / relative)
        loaded[relative] = value
        source_sha256s[relative] = digest

    receipt = loaded["topic-job-receipt.json"]
    if receipt.get("status") != "complete" or receipt.get("topic_id") != topic_id:
        raise ValueError("topic job receipt is not complete for the requested topic")

    decomposition = loaded["decomposition/result.json"]
    if decomposition.get("error") is not None:
        raise ValueError("topic decomposition contains an error")
    topic = decomposition.get("topic")
    if not isinstance(topic, dict) or topic.get("id") != topic_id:
        raise ValueError("decomposition topic identity does not match")
    narrative = _required_text(topic.get("narrative"), label="topic narrative")
    raw_subnarratives = decomposition.get("subnarratives")
    if not isinstance(raw_subnarratives, list) or not raw_subnarratives:
        raise ValueError("at least one valid subnarrative is required")
    subnarratives: list[SemanticUnit] = []
    for raw in raw_subnarratives:
        if not isinstance(raw, dict) or raw.get("topic_id") != topic_id:
            raise ValueError("subnarrative topic identity does not match")
        subnarrative_id = _required_text(
            raw.get("subnarrative_id"), label="subnarrative ID"
        )
        text = _required_text(raw.get("text"), label="subnarrative text")
        raw_queries = raw.get("bm25_queries")
        if not isinstance(raw_queries, list) or any(
            not isinstance(query, str) or not query.strip() for query in raw_queries
        ):
            raise ValueError("subnarrative BM25 query list is invalid")
        subnarratives.append(
            SemanticUnit(
                subnarrative_id=subnarrative_id,
                text=text,
                retrieval_query_texts=tuple(raw_queries),
            )
        )
    subnarrative_ids = {row.subnarrative_id for row in subnarratives}
    if len(subnarrative_ids) != len(subnarratives):
        raise ValueError("subnarrative IDs must be unique")

    audit = loaded["retrieval/audit.json"]
    if audit.get("topic_id") != topic_id or not isinstance(audit.get("lanes"), list):
        raise ValueError("retrieval audit topic identity or lanes are invalid")
    expected_lanes = {"original"} | {
        f"facet:{subnarrative_id}:text" for subnarrative_id in subnarrative_ids
    }
    seen_lanes: set[str] = set()
    document_bindings: dict[str, tuple[str, int]] = {}
    subnarrative_source_ranks: dict[str, dict[str, int]] = {}
    for raw_lane in audit["lanes"]:
        if not isinstance(raw_lane, dict):
            raise ValueError("retrieval audit lane must be an object")
        lane_name = _required_text(raw_lane.get("lane_name"), label="lane name")
        if lane_name in seen_lanes:
            raise ValueError("retrieval audit lane names must be unique")
        seen_lanes.add(lane_name)
        raw_candidates = raw_lane.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError("retrieval audit candidates must be a list")
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                raise ValueError("retrieval candidate must be an object")
            docid = _required_text(raw_candidate.get("docid"), label="document ID")
            content_digest = _required_text(
                raw_candidate.get("text_sha256"), label="document SHA-256"
            )
            if not re.fullmatch(r"[0-9a-f]{64}", content_digest):
                raise ValueError("document SHA-256 must be lowercase hexadecimal")
            rank = raw_candidate.get("bm25_rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                raise ValueError("retrieval rank must be a positive integer")
            prior = document_bindings.get(docid)
            if prior is not None and prior[0] != content_digest:
                raise ValueError("one document ID is bound to conflicting content")
            document_bindings[docid] = (
                content_digest,
                rank if prior is None else min(rank, prior[1]),
            )
            if lane_name != "original":
                subnarrative_id = lane_name.removeprefix("facet:").removesuffix(
                    ":text"
                )
                ranks = subnarrative_source_ranks.setdefault(docid, {})
                ranks[subnarrative_id] = min(rank, ranks.get(subnarrative_id, rank))
    if seen_lanes != expected_lanes:
        raise ValueError("retrieval audit does not contain the exact authenticated lanes")

    selection = loaded["scoring/selection.json"]
    if selection.get("topic_id") != topic_id or not isinstance(
        selection.get("union_pool"), list
    ):
        raise ValueError("selection union pool is invalid")
    union_docids: list[str] = []
    for raw in selection["union_pool"]:
        if not isinstance(raw, dict):
            raise ValueError("selection union pool row must be an object")
        union_docids.append(_required_text(raw.get("docid"), label="union document ID"))
    if len(set(union_docids)) != len(union_docids):
        raise ValueError("selection union pool repeats a document")
    if set(union_docids) != set(document_bindings):
        raise ValueError("selection union pool differs from retrieval audit documents")

    store = DocumentStore(Path(document_store_root))
    documents: list[SourceDocument] = []
    for docid in sorted(union_docids, key=lambda value: value.encode("utf-8")):
        content_digest, rank = document_bindings[docid]
        text = store.read_text(content_digest)
        if not text.strip():
            raise ValueError("source document text must be nonempty")
        documents.append(
            SourceDocument(
                docid=docid,
                content_sha256=content_digest,
                best_retrieval_rank=rank,
                subnarrative_source_ranks=dict(
                    sorted(subnarrative_source_ranks.get(docid, {}).items())
                ),
                text=text,
            )
        )
    return TopicInput(
        topic_id=topic_id,
        narrative=narrative,
        subnarratives=tuple(subnarratives),
        documents=tuple(documents),
        source_sha256s=source_sha256s,
    )


def _counter_snapshot(scorer: PassageScorer) -> dict[str, int]:
    raw = scorer.stats
    if not isinstance(raw, Mapping):
        raise TypeError("passage scorer stats must be a mapping")
    result: dict[str, int] = {}
    for name in ("cache_hits", "cache_misses", "model_batches"):
        value = raw.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("passage scorer counters must be nonnegative integers")
        result[name] = value
    return result


def score_topic(
    topic: TopicInput,
    *,
    scorer: PassageScorer,
    chunker: SemanticTextChunker,
) -> TopicMatrix:
    """Score the full union against the narrative and each pooled subnarrative."""

    if not isinstance(topic, TopicInput):
        raise TypeError("topic must be a TopicInput")
    if not topic.documents or not topic.subnarratives:
        raise ValueError("topic must contain documents and subnarratives")
    if not isinstance(chunker, SemanticTextChunker):
        raise TypeError("chunker must be a SemanticTextChunker")
    identity = scorer.identity
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("passage scorer identity must be a nonempty mapping")
    scorer_identity = json.loads(
        json.dumps(dict(identity), sort_keys=True, separators=(",", ":"), allow_nan=False)
    )

    all_chunks: list[TextChunk] = []
    for document in topic.documents:
        chunks = tuple(chunker.split_text(document.text, document_id=document.docid))
        if not chunks:
            raise ValueError("every source document must yield at least one passage")
        for chunk in chunks:
            if (
                chunk.document_id != document.docid
                or chunk.start_char < 0
                or chunk.end_char <= chunk.start_char
                or document.text[chunk.start_char : chunk.end_char] != chunk.text
            ):
                raise ValueError("chunker returned a passage outside its source document")
        all_chunks.extend(chunks)
    chunk_rows = tuple(all_chunks)

    units = (("__narrative__", "narrative", topic.narrative),) + tuple(
        (row.subnarrative_id, "subnarrative", row.text)
        for row in topic.subnarratives
    )
    if len({row[0] for row in units}) != len(units):
        raise ValueError("matrix semantic unit IDs must be unique")

    before = _counter_snapshot(scorer)
    passage_rows: list[MatrixPassage] = []
    for unit_id, _kind, query_text in units:
        scored = tuple(scorer.rank(query_text, chunk_rows))
        if len(scored) != len(chunk_rows):
            raise ValueError("passage scorer returned an incomplete matrix row")
        by_chunk = {row.chunk.chunk_id: row for row in scored}
        expected_chunk_ids = {chunk.chunk_id for chunk in chunk_rows}
        if len(by_chunk) != len(chunk_rows) or set(by_chunk) != expected_chunk_ids:
            raise ValueError("passage scorer changed the complete chunk identity set")
        for chunk in chunk_rows:
            scored_row = by_chunk[chunk.chunk_id]
            if scored_row.chunk != chunk:
                raise ValueError("passage scorer changed chunk content or span binding")
            finite = PassageScore(
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                raw_score=scored_row.relevance_score,
            )
            passage_rows.append(
                MatrixPassage(
                    unit_id=unit_id,
                    docid=chunk.document_id,
                    start_char=finite.start_char,
                    end_char=finite.end_char,
                    raw_score=float(finite.raw_score),
                )
            )
    after = _counter_snapshot(scorer)
    stats = {name: after[name] - before[name] for name in before}
    if any(value < 0 for value in stats.values()):
        raise ValueError("passage scorer counters decreased during topic scoring")

    config = chunker.config
    return TopicMatrix(
        topic_id=topic.topic_id,
        documents=tuple(
            MatrixDocument(
                docid=row.docid,
                content_sha256=row.content_sha256,
                best_retrieval_rank=row.best_retrieval_rank,
                subnarrative_source_ranks=dict(row.subnarrative_source_ranks),
            )
            for row in topic.documents
        ),
        units=tuple(
            MatrixUnit(
                unit_id=unit_id,
                kind=kind,
                text_sha256=sha256(query_text.encode("utf-8")).hexdigest(),
            )
            for unit_id, kind, query_text in units
        ),
        passages=tuple(
            sorted(
                passage_rows,
                key=lambda row: (
                    row.unit_id.encode("utf-8"),
                    row.docid.encode("utf-8"),
                    row.start_char,
                    row.end_char,
                ),
            )
        ),
        source_sha256s=dict(topic.source_sha256s),
        scorer_identity=scorer_identity,
        chunker_identity={
            "backend": "trec_rag.chunking.SemanticTextChunker",
            "max_characters": config.max_characters,
            "overlap_characters": config.overlap_characters,
            "trim": config.trim,
        },
        cache_stats=stats,
    )


def _canonical_json_line(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _matrix_bytes(matrix: TopicMatrix) -> bytes:
    rows: list[dict[str, object]] = [
        {
            "record_type": "header",
            "schema_version": "retrieval-baseline-topic-matrix-v1",
            "topic_id": matrix.topic_id,
            "source_sha256s": dict(matrix.source_sha256s),
            "scorer_identity": dict(matrix.scorer_identity),
            "chunker_identity": dict(matrix.chunker_identity),
            "cache_stats": dict(matrix.cache_stats),
        }
    ]
    rows.extend(
        {
            "record_type": "document",
            "docid": row.docid,
            "content_sha256": row.content_sha256,
            "best_retrieval_rank": row.best_retrieval_rank,
            "subnarrative_source_ranks": dict(row.subnarrative_source_ranks),
        }
        for row in matrix.documents
    )
    rows.extend(
        {
            "record_type": "unit",
            "unit_id": row.unit_id,
            "kind": row.kind,
            "text_sha256": row.text_sha256,
        }
        for row in matrix.units
    )
    rows.extend(
        {
            "record_type": "passage",
            "unit_id": row.unit_id,
            "docid": row.docid,
            "start_char": row.start_char,
            "end_char": row.end_char,
            "raw_score": row.raw_score,
        }
        for row in matrix.passages
    )
    return b"".join(_canonical_json_line(row) for row in rows)


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_topic_matrix(matrix: TopicMatrix, output_dir: Path) -> Path:
    """Atomically write one canonical text-free matrix and digest manifest."""

    root = Path(output_dir)
    matrix_path = root / "topic-matrix.jsonl"
    body = _matrix_bytes(matrix)
    _atomic_write(matrix_path, body)
    manifest = {
        "schema_version": "retrieval-baseline-topic-matrix-manifest-v1",
        "topic_id": matrix.topic_id,
        "matrix_sha256": sha256(body).hexdigest(),
        "matrix_byte_count": len(body),
        "document_count": len(matrix.documents),
        "semantic_unit_count": len(matrix.units),
        "passage_pair_count": len(matrix.passages),
    }
    manifest_path = root / "topic-matrix-manifest.json"
    _atomic_write(manifest_path, _canonical_json_line(manifest))
    return manifest_path


def _require_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def read_topic_matrix(output_dir: Path) -> TopicMatrix:
    """Read and authenticate one canonical topic matrix."""

    root = Path(output_dir)
    body = (root / "topic-matrix.jsonl").read_bytes()
    manifest = _require_mapping(
        json.loads((root / "topic-matrix-manifest.json").read_bytes()),
        label="topic matrix manifest",
    )
    if manifest.get("schema_version") != "retrieval-baseline-topic-matrix-manifest-v1":
        raise ValueError("topic matrix manifest schema is invalid")
    if manifest.get("matrix_sha256") != sha256(body).hexdigest():
        raise ValueError("topic matrix digest does not match its manifest")
    parsed = [
        _require_mapping(json.loads(line), label="topic matrix row")
        for line in body.splitlines()
        if line
    ]
    if not parsed or parsed[0].get("record_type") != "header":
        raise ValueError("topic matrix header is missing")
    header = parsed[0]
    if header.get("schema_version") != "retrieval-baseline-topic-matrix-v1":
        raise ValueError("topic matrix schema is invalid")
    documents: list[MatrixDocument] = []
    units: list[MatrixUnit] = []
    passages: list[MatrixPassage] = []
    for row in parsed[1:]:
        record_type = row.get("record_type")
        if record_type == "document":
            documents.append(
                MatrixDocument(
                    docid=str(row["docid"]),
                    content_sha256=str(row["content_sha256"]),
                    best_retrieval_rank=int(row["best_retrieval_rank"]),
                    subnarrative_source_ranks={
                        str(name): int(rank)
                        for name, rank in _require_mapping(
                            row.get("subnarrative_source_ranks"),
                            label="subnarrative source ranks",
                        ).items()
                    },
                )
            )
        elif record_type == "unit":
            units.append(
                MatrixUnit(
                    unit_id=str(row["unit_id"]),
                    kind=str(row["kind"]),
                    text_sha256=str(row["text_sha256"]),
                )
            )
        elif record_type == "passage":
            passages.append(
                MatrixPassage(
                    unit_id=str(row["unit_id"]),
                    docid=str(row["docid"]),
                    start_char=int(row["start_char"]),
                    end_char=int(row["end_char"]),
                    raw_score=float(row["raw_score"]),
                )
            )
        else:
            raise ValueError("topic matrix contains an unknown record type")
    matrix = TopicMatrix(
        topic_id=str(header["topic_id"]),
        documents=tuple(documents),
        units=tuple(units),
        passages=tuple(passages),
        source_sha256s=_require_mapping(
            header.get("source_sha256s"), label="source hashes"
        ),
        scorer_identity=_require_mapping(
            header.get("scorer_identity"), label="scorer identity"
        ),
        chunker_identity=_require_mapping(
            header.get("chunker_identity"), label="chunker identity"
        ),
        cache_stats={
            str(name): int(value)
            for name, value in _require_mapping(
                header.get("cache_stats"), label="cache stats"
            ).items()
        },
    )
    expected_counts = {
        "document_count": len(matrix.documents),
        "semantic_unit_count": len(matrix.units),
        "passage_pair_count": len(matrix.passages),
    }
    if any(manifest.get(name) != count for name, count in expected_counts.items()):
        raise ValueError("topic matrix manifest counts do not match")
    if manifest.get("topic_id") != matrix.topic_id:
        raise ValueError("topic matrix topic identity does not match manifest")
    if _matrix_bytes(matrix) != body:
        raise ValueError("topic matrix rows are not in canonical order")
    return matrix


def suppress_overlaps(passages: tuple[PassageScore, ...]) -> tuple[PassageScore, ...]:
    """Greedily retain high-scoring spans whose overlap coefficient is below 0.5."""

    ordered = sorted(
        passages,
        key=lambda row: (-row.raw_score, row.start_char, row.end_char),
    )
    retained: list[PassageScore] = []
    for candidate in ordered:
        if all(_overlap_coefficient(candidate, prior) < 0.5 for prior in retained):
            retained.append(candidate)
    return tuple(retained)


def weighted_passage_score(passages: tuple[PassageScore, ...]) -> float:
    """Aggregate up to four overlap-suppressed passage logits."""

    retained = suppress_overlaps(passages)[: len(_TOP_PASSAGE_WEIGHTS)]
    if not retained:
        raise ValueError("at least one passage score is required")
    weights = _TOP_PASSAGE_WEIGHTS[: len(retained)]
    denominator = sum(weights)
    return sum(
        weight * passage.raw_score
        for weight, passage in zip(weights, retained, strict=True)
    ) / denominator


def midrank_percentiles(values: Mapping[_Key, float]) -> dict[_Key, float]:
    """Return deterministic tie-aware percentiles for one topic/semantic unit."""

    items = list(values.items())
    for _key, value in items:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("percentile values must be finite real numbers")
        if not math.isfinite(value):
            raise ValueError("percentile values must be finite")
    if not items:
        return {}
    if len(items) == 1:
        return {items[0][0]: 1.0}

    ordered_values = sorted(value for _key, value in items)
    by_value: dict[float, float] = {}
    lower_count = 0
    while lower_count < len(ordered_values):
        value = ordered_values[lower_count]
        tied_count = 1
        while (
            lower_count + tied_count < len(ordered_values)
            and ordered_values[lower_count + tied_count] == value
        ):
            tied_count += 1
        by_value[value] = (
            lower_count + (tied_count - 1) / 2
        ) / (len(ordered_values) - 1)
        lower_count += tied_count
    return {key: by_value[value] for key, value in items}


def _document_index(documents: tuple[DocumentScore, ...]) -> dict[str, DocumentScore]:
    if not documents:
        raise ValueError("at least one document score is required")
    by_docid = {document.docid: document for document in documents}
    if len(by_docid) != len(documents):
        raise ValueError("document scores must have unique docids")
    subnarratives = set(documents[0].subnarrative_percentiles)
    if any(set(document.subnarrative_percentiles) != subnarratives for document in documents):
        raise ValueError("documents must share one complete subnarrative set")
    return by_docid


def _docids_sha256(docids: tuple[str, ...]) -> str:
    body = b"".join(docid.encode("utf-8") + b"\n" for docid in docids)
    return sha256(body).hexdigest()


def cutoff_decision(documents: tuple[DocumentScore, ...]) -> CutoffDecision:
    """Apply raw per-unit robust thresholds, union them, and record diagnostics."""

    _document_index(documents)
    subnarrative_ids = tuple(documents[0].subnarrative_raw_scores)
    unit_scores = (
        ("__narrative__", {row.docid: row.narrative_raw_score for row in documents}),
    ) + tuple(
        (
            subnarrative_id,
            {
                row.docid: row.subnarrative_raw_scores[subnarrative_id]
                for row in documents
            },
        )
        for subnarrative_id in subnarrative_ids
    )
    eligible_set: set[str] = set()
    unit_stats: list[CutoffUnitStat] = []
    for unit_id, scores in unit_scores:
        center = median(scores.values())
        deviation = median(abs(value - center) for value in scores.values())
        if deviation > 0.0:
            threshold = center + 2.5 * 1.4826 * deviation
            admitted = tuple(
                sorted(
                    (docid for docid, value in scores.items() if value >= threshold),
                    key=lambda docid: docid.encode("utf-8"),
                )
            )
            comparison = "greater_than_or_equal"
        else:
            threshold = center
            admitted = tuple(
                sorted(
                    (docid for docid, value in scores.items() if value > center),
                    key=lambda docid: docid.encode("utf-8"),
                )
            )
            comparison = "strictly_greater_than"
        eligible_set.update(admitted)
        unit_stats.append(
            CutoffUnitStat(
                unit_id=unit_id,
                median=float(center),
                mad=float(deviation),
                threshold=float(threshold),
                comparison=comparison,
                admitted_count=len(admitted),
                admitted_docids_sha256=_docids_sha256(admitted),
            )
        )
    pre_fallback_count = len(eligible_set)
    fallback_used = not eligible_set
    if fallback_used:
        eligible_set.add(
            min(
                documents,
                key=lambda document: (
                    -document.narrative_raw_score,
                    document.best_retrieval_rank,
                    document.docid.encode("utf-8"),
                ),
            ).docid
        )
    eligible = tuple(
        sorted(eligible_set, key=lambda docid: docid.encode("utf-8"))
    )
    return CutoffDecision(
        units=tuple(unit_stats),
        eligible_docids=eligible,
        pre_fallback_count=pre_fallback_count,
        fallback_used=fallback_used,
    )


def select_eligible_documents(documents: tuple[DocumentScore, ...]) -> tuple[str, ...]:
    """Return the shared topic-specific eligible set."""

    return cutoff_decision(documents).eligible_docids


def breadth_counts(
    documents: tuple[DocumentScore, ...],
    hits: tuple[BreadthPassage, ...],
) -> dict[str, tuple[int, int]]:
    """Count top passage evidence after per-document then per-subnarrative limits."""

    by_docid = _document_index(documents)
    subnarratives = set(next(iter(by_docid.values())).subnarrative_percentiles)
    grouped: dict[tuple[str, str], list[PassageScore]] = {}
    for hit in hits:
        if hit.docid not in by_docid:
            raise ValueError("breadth hit refers to a document outside the topic union")
        if hit.subnarrative_id not in subnarratives:
            raise ValueError("breadth hit refers to an unknown subnarrative")
        if hit.subnarrative_id not in by_docid[hit.docid].subnarrative_source_ranks:
            raise ValueError("breadth hit refers to a document outside the pooled source")
        grouped.setdefault((hit.subnarrative_id, hit.docid), []).append(hit.passage)

    per_subnarrative: dict[str, list[tuple[str, PassageScore]]] = {
        subnarrative_id: [] for subnarrative_id in subnarratives
    }
    for (subnarrative_id, docid), passages in grouped.items():
        for passage in suppress_overlaps(tuple(passages))[:3]:
            per_subnarrative[subnarrative_id].append((docid, passage))

    admitted: dict[str, list[str]] = {docid: [] for docid in by_docid}
    for subnarrative_id, candidates in per_subnarrative.items():
        candidates.sort(
            key=lambda row: (
                -row[1].raw_score,
                by_docid[row[0]].subnarrative_source_ranks[subnarrative_id],
                row[0].encode("utf-8"),
                row[1].start_char,
                row[1].end_char,
            )
        )
        for docid, _passage in candidates[:100]:
            admitted[docid].append(subnarrative_id)
    return {
        docid: (len(set(subnarrative_ids)), len(subnarrative_ids))
        for docid, subnarrative_ids in admitted.items()
    }


def rank_topic_matrix(matrix: TopicMatrix) -> RankedTopic:
    """Aggregate one authenticated complete matrix and build all three rankings."""

    if not isinstance(matrix, TopicMatrix):
        raise TypeError("matrix must be a TopicMatrix")
    documents = {row.docid: row for row in matrix.documents}
    if not documents or len(documents) != len(matrix.documents):
        raise ValueError("matrix documents must be nonempty and unique")
    units = {row.unit_id: row for row in matrix.units}
    if not units or len(units) != len(matrix.units):
        raise ValueError("matrix semantic units must be nonempty and unique")
    narrative_units = [row.unit_id for row in matrix.units if row.kind == "narrative"]
    subnarrative_ids = tuple(
        row.unit_id for row in matrix.units if row.kind == "subnarrative"
    )
    if narrative_units != ["__narrative__"] or not subnarrative_ids:
        raise ValueError("matrix must contain one narrative and valid subnarratives")
    if any(row.kind not in {"narrative", "subnarrative"} for row in matrix.units):
        raise ValueError("matrix semantic unit kind is invalid")
    if any(
        not set(document.subnarrative_source_ranks).issubset(subnarrative_ids)
        for document in matrix.documents
    ):
        raise ValueError("matrix pooled-source rank refers to an unknown subnarrative")

    grouped: dict[tuple[str, str], list[PassageScore]] = {}
    for row in matrix.passages:
        if row.unit_id not in units or row.docid not in documents:
            raise ValueError("matrix passage refers to an unknown unit or document")
        grouped.setdefault((row.unit_id, row.docid), []).append(
            PassageScore(row.start_char, row.end_char, row.raw_score)
        )
    raw_by_unit: dict[str, dict[str, float]] = {}
    for unit_id in units:
        raw_by_unit[unit_id] = {}
        for docid in documents:
            passages = grouped.get((unit_id, docid))
            if not passages:
                raise ValueError("matrix is missing a document/semantic-unit score")
            raw_by_unit[unit_id][docid] = weighted_passage_score(tuple(passages))
    percentiles = {
        unit_id: midrank_percentiles(scores)
        for unit_id, scores in raw_by_unit.items()
    }
    document_scores = tuple(
        DocumentScore(
            docid=document.docid,
            best_retrieval_rank=document.best_retrieval_rank,
            narrative_raw_score=raw_by_unit["__narrative__"][document.docid],
            subnarrative_raw_scores={
                unit_id: raw_by_unit[unit_id][document.docid]
                for unit_id in subnarrative_ids
            },
            narrative_percentile=percentiles["__narrative__"][document.docid],
            subnarrative_percentiles={
                unit_id: percentiles[unit_id][document.docid]
                for unit_id in subnarrative_ids
            },
            subnarrative_source_ranks=dict(document.subnarrative_source_ranks),
        )
        for document in matrix.documents
    )
    breadth_passages = tuple(
        BreadthPassage(
            subnarrative_id=row.unit_id,
            docid=row.docid,
            passage=PassageScore(row.start_char, row.end_char, row.raw_score),
        )
        for row in matrix.passages
        if row.unit_id in subnarrative_ids
        and row.unit_id in documents[row.docid].subnarrative_source_ranks
    )
    cutoff = cutoff_decision(document_scores)
    rankings = build_rankings(document_scores, breadth_passages)
    if rankings.eligible_docids != cutoff.eligible_docids:
        raise ValueError("ranking eligible set differs from cutoff decision")
    return RankedTopic(
        topic_id=matrix.topic_id,
        document_scores=document_scores,
        breadth_passages=breadth_passages,
        cutoff=cutoff,
        rankings=rankings,
    )


def build_rankings(
    documents: tuple[DocumentScore, ...],
    hits: tuple[BreadthPassage, ...],
) -> RunRankings:
    """Build three orderings over one shared eligible document set."""

    by_docid = _document_index(documents)
    eligible = select_eligible_documents(documents)
    counts = breadth_counts(documents, hits)

    narrative = tuple(
        sorted(
            eligible,
            key=lambda docid: (
                -by_docid[docid].narrative_percentile,
                by_docid[docid].best_retrieval_rank,
                docid.encode("utf-8"),
            ),
        )
    )
    combo = tuple(
        sorted(
            eligible,
            key=lambda docid: (
                -by_docid[docid].combo_score,
                by_docid[docid].best_retrieval_rank,
                docid.encode("utf-8"),
            ),
        )
    )
    breadth = tuple(
        sorted(
            eligible,
            key=lambda docid: (
                -counts[docid][0],
                -counts[docid][1],
                -by_docid[docid].combo_score,
                by_docid[docid].best_retrieval_rank,
                docid.encode("utf-8"),
            ),
        )
    )
    return RunRankings(
        eligible_docids=eligible,
        narrative=narrative,
        combo=combo,
        breadth=breadth,
    )


_RUN_IDS = {
    "narrative": "r26-narrative-v1",
    "combo": "r26-narrative-facet-v1",
    "breadth": "r26-facet-breadth-v1",
}


def _validate_trec_token(value: str, *, label: str) -> None:
    if not value or any(character.isspace() for character in value):
        raise ValueError(f"{label} must be a nonempty token without whitespace")


def export_runs(matrices: tuple[TopicMatrix, ...], output_dir: Path) -> Path:
    """Write three deterministic organizer-facing run files and one manifest."""

    if not matrices:
        raise ValueError("at least one topic matrix is required")
    by_topic = {matrix.topic_id: matrix for matrix in matrices}
    if len(by_topic) != len(matrices):
        raise ValueError("topic matrices must have unique topic IDs")
    ordered = tuple(sorted(matrices, key=lambda row: topic_sort_key(row.topic_id)))
    if any(matrix.scorer_identity != ordered[0].scorer_identity for matrix in ordered):
        raise ValueError("topic matrices must share one scorer identity")
    if any(matrix.chunker_identity != ordered[0].chunker_identity for matrix in ordered):
        raise ValueError("topic matrices must share one chunker identity")

    ranked_topics = tuple(rank_topic_matrix(matrix) for matrix in ordered)
    root = Path(output_dir)
    run_files = {
        "narrative": "retrieval-narrative.run",
        "combo": "retrieval-narrative-facet.run",
        "breadth": "retrieval-facet-breadth.run",
    }
    for run_name, relative_path in run_files.items():
        run_id = _RUN_IDS[run_name]
        _validate_trec_token(run_id, label="run ID")
        lines: list[str] = []
        for ranked in ranked_topics:
            _validate_trec_token(ranked.topic_id, label="topic ID")
            ranking = getattr(ranked.rankings, run_name)
            depth = len(ranking)
            for rank, docid in enumerate(ranking, start=1):
                _validate_trec_token(docid, label="document ID")
                lines.append(
                    f"{ranked.topic_id} Q0 {docid} {rank} {depth - rank + 1} {run_id}\n"
                )
        _atomic_write(root / relative_path, "".join(lines).encode("utf-8"))

    topics: list[dict[str, object]] = []
    for matrix, ranked in zip(ordered, ranked_topics, strict=True):
        eligible_hash = _docids_sha256(ranked.rankings.eligible_docids)
        topics.append(
            {
                "topic_id": matrix.topic_id,
                "document_union_count": len(matrix.documents),
                "semantic_unit_count": len(matrix.units),
                "passage_pair_count": len(matrix.passages),
                "matrix_sha256": sha256(_matrix_bytes(matrix)).hexdigest(),
                "k": len(ranked.rankings.eligible_docids),
                "pre_fallback_count": ranked.cutoff.pre_fallback_count,
                "fallback_used": ranked.cutoff.fallback_used,
                "cutoff_units": [asdict(row) for row in ranked.cutoff.units],
                "narrative_docids_sha256": eligible_hash,
                "combo_docids_sha256": eligible_hash,
                "breadth_docids_sha256": eligible_hash,
            }
        )
    manifest = {
        "schema_version": "retrieval-baseline-runs-manifest-v1",
        "run_files": run_files,
        "run_ids": _RUN_IDS,
        "scorer_identity": dict(ordered[0].scorer_identity),
        "chunker_identity": dict(ordered[0].chunker_identity),
        "topics": topics,
    }
    manifest_path = root / "retrieval-baseline-runs-manifest.json"
    _atomic_write(manifest_path, _canonical_json_line(manifest))
    return manifest_path


__all__ = [
    "BreadthPassage",
    "CutoffDecision",
    "CutoffUnitStat",
    "DocumentScore",
    "MatrixDocument",
    "MatrixPassage",
    "MatrixUnit",
    "PassageScore",
    "RunRankings",
    "RankedTopic",
    "SemanticUnit",
    "SourceDocument",
    "TopicInput",
    "TopicMatrix",
    "breadth_counts",
    "build_rankings",
    "cutoff_decision",
    "export_runs",
    "load_topic_input",
    "midrank_percentiles",
    "read_topic_matrix",
    "rank_topic_matrix",
    "score_topic",
    "select_eligible_documents",
    "suppress_overlaps",
    "topic_sort_key",
    "weighted_passage_score",
    "write_topic_matrix",
]
