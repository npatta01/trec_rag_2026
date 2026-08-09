"""Stable chunking contracts for reranking and evidence selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Protocol


class MissingChunkerDependency(RuntimeError):
    """Raised when an optional chunking backend is not installed."""


class MissingSegmenterDependency(RuntimeError):
    """Raised when the sentence-segmentation backend is not installed."""


@dataclass(frozen=True)
class ChunkingConfig:
    max_characters: int = 12_000
    overlap_characters: int = 800
    trim: bool = True

    def __post_init__(self) -> None:
        if self.max_characters <= 0:
            raise ValueError("max_characters must be positive")
        if self.overlap_characters < 0:
            raise ValueError("overlap_characters must be non-negative")
        if self.overlap_characters >= self.max_characters:
            raise ValueError("overlap_characters must be smaller than max_characters")


@dataclass(frozen=True)
class TextChunk:
    document_id: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int


class TextChunker(Protocol):
    def split_text(self, text: str, *, document_id: str) -> list[TextChunk]:
        ...


SPACY_SEGMENTER_MODEL = "en_core_web_sm"
# The parser supplies sentence boundaries. The tagger and attribute_ruler
# together supply `token.pos_`, which is how a heading is told from a sentence:
# the tagger sets the fine-grained `tag_` and attribute_ruler maps it to the
# coarse `pos_`. Dropping attribute_ruler leaves `pos_` empty on every token and
# silently reduces completeness to a punctuation check, so it stays enabled.
_SPACY_DISABLED_PIPES = ("ner", "lemmatizer")
_TERMINAL_PUNCTUATION = re.compile(r"[.!?][\"'”’)\]}]*$")
_BLANK_LINE = re.compile(r"\n[^\S\n]*\n")
# Well inside spaCy's 1,000,000-character ceiling, because the parser's memory
# grows with document length and this process also shares memory with the GPU.
SPACY_MAX_TEXT_CHARACTERS = 100_000


@dataclass(frozen=True)
class Sentence:
    """One sentence as an exact half-open character range of its source."""

    start_char: int
    end_char: int
    is_complete: bool

    def __post_init__(self) -> None:
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError("sentence span must be a non-empty range")


class SentenceSegmenter(Protocol):
    identity: Mapping[str, object]

    def segment(self, text: str) -> list[Sentence]:
        ...


def _import_spacy():
    try:
        import spacy
    except ImportError as exc:
        raise MissingSegmenterDependency(
            "spacy is required for SpacySentenceSegmenter. "
            "Run code/tools/setup_env.sh, then use .venv/bin/python."
        ) from exc
    return spacy


class SpacySentenceSegmenter:
    """Sentence segmentation with exact character offsets.

    Offsets index the untouched source, so a caller can slice the original text
    and get back byte-identical sentences. That is what lets extraction stay
    provenance-complete without this module owning any span type of its own.

    ``is_complete`` marks whether a sentence stands on its own. A run of words
    with no finite verb and no terminal punctuation - a heading, a nav label, a
    list caption - does not, and callers join it to what follows rather than
    admitting it as standalone evidence.
    """

    def __init__(self, model: str = SPACY_SEGMENTER_MODEL) -> None:
        self.model = model
        self._nlp: Any | None = None

    @property
    def identity(self) -> Mapping[str, object]:
        return {
            "backend": "spacy",
            "model": self.model,
            "model_version": self._model_version(),
            "disabled": list(_SPACY_DISABLED_PIPES),
        }

    def _model_version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(self.model)
        except PackageNotFoundError:  # pragma: no cover - model always pinned
            raise MissingSegmenterDependency(
                f"spaCy model {self.model} is not installed"
            ) from None

    def _get_pipeline(self) -> Any:
        if self._nlp is None:
            spacy = _import_spacy()
            try:
                self._nlp = spacy.load(self.model, exclude=list(_SPACY_DISABLED_PIPES))
            except OSError as exc:
                raise MissingSegmenterDependency(
                    f"spaCy model {self.model} is not installed"
                ) from exc
        return self._nlp

    def segment(self, text: str) -> list[Sentence]:
        if not text.strip():
            return []
        sentences: list[Sentence] = []
        for block_start, block_end in _bounded_blocks(text, SPACY_MAX_TEXT_CHARACTERS):
            sentences.extend(self._segment_block(text, block_start, block_end))
        return sentences

    def _segment_block(self, text: str, block_start: int, block_end: int) -> list[Sentence]:
        doc = self._get_pipeline()(text[block_start:block_end])
        sentences: list[Sentence] = []
        for sent in doc.sents:
            if not sent.text.strip():
                continue
            span_start = block_start + sent.start_char
            span_end = block_start + sent.end_char
            for start, end in _split_on_blank_lines(text, span_start, span_end):
                # token.idx is block-relative; start/end are whole-document.
                tokens = [
                    token
                    for token in sent
                    if block_start + token.idx >= start
                    and block_start + token.idx + len(token.text) <= end
                ]
                sentences.append(
                    Sentence(
                        start_char=start,
                        end_char=end,
                        is_complete=_is_complete(text[start:end], tokens),
                    )
                )
        return sentences


def _bounded_blocks(text: str, limit: int) -> list[tuple[int, int]]:
    """Cover the text with ranges the parser can hold, preferring blank lines.

    spaCy refuses input past ``nlp.max_length`` because the parser's memory grows
    with document length, and the corpus does contain documents past a million
    characters. Splitting there rather than raising keeps one long document from
    aborting a topic after hours of scoring.

    Ranges tile the text exactly and are returned in source order, so shifting a
    block-relative offset by its start yields a whole-document offset. Blank
    lines are preferred cut points; text with none is cut on length alone.
    """
    if len(text) <= limit:
        return [(0, len(text))]
    blocks: list[tuple[int, int]] = []
    start = 0
    while len(text) - start > limit:
        window_end = start + limit
        cut = 0
        for match in _BLANK_LINE.finditer(text, start, window_end):
            cut = match.end()
        if cut <= start:
            # No blank line in reach; fall back to a whitespace boundary so a
            # block edge does not land inside a word.
            cut = text.rfind(" ", start, window_end)
            if cut <= start:
                cut = window_end
        blocks.append((start, cut))
        start = cut
    blocks.append((start, len(text)))
    return blocks


def _split_on_blank_lines(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Cut one span at blank lines and trim each piece to non-whitespace.

    A blank line separates blocks whatever the model decided, so a missed
    sentence boundary can never silently merge two unrelated sections. Trimming
    keeps candidate text free of stray whitespace, and narrowing a range only
    ever keeps offsets exact.
    """
    pieces: list[tuple[int, int]] = []
    cursor = start
    for match in _BLANK_LINE.finditer(text, start, end):
        pieces.append((cursor, match.start()))
        cursor = match.end()
    pieces.append((cursor, end))
    trimmed: list[tuple[int, int]] = []
    for piece_start, piece_end in pieces:
        while piece_start < piece_end and text[piece_start].isspace():
            piece_start += 1
        while piece_end > piece_start and text[piece_end - 1].isspace():
            piece_end -= 1
        if piece_start < piece_end:
            trimmed.append((piece_start, piece_end))
    return trimmed


@lru_cache(maxsize=1)
def sentence_segmenter() -> SentenceSegmenter:
    """The process-wide segmenter, so every lane cuts text identically.

    Loading the model is the expensive part, and two lanes disagreeing about
    where a sentence ends is exactly the drift this contract exists to prevent.
    """
    return SpacySentenceSegmenter()


def _is_complete(text: str, tokens: Sequence[Any]) -> bool:
    """A sentence stands alone when it is punctuated or carries a finite verb."""
    stripped = text.strip()
    if not stripped:
        return False
    if _TERMINAL_PUNCTUATION.search(stripped):
        return True
    return any("Fin" in token.morph.get("VerbForm") for token in tokens)


def _import_text_splitter():
    try:
        from semantic_text_splitter import TextSplitter
    except ImportError as exc:
        raise MissingChunkerDependency(
            "semantic-text-splitter is required for SemanticTextChunker. "
            "Run code/tools/setup_env.sh, then use .venv/bin/python."
        ) from exc
    return TextSplitter


class SemanticTextChunker:
    """Text chunker backed by semantic-text-splitter.

    Keep callers on the TextChunker contract so the backend can later change
    without changing reranking or evidence-selection code.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self._splitter: Any | None = None

    def _get_splitter(self) -> Any:
        if self._splitter is None:
            self._splitter = _import_text_splitter()(
                self.config.max_characters,
                overlap=self.config.overlap_characters,
                trim=self.config.trim,
            )
        return self._splitter

    def split_text(self, text: str, *, document_id: str) -> list[TextChunk]:
        if not document_id.strip():
            raise ValueError("document_id is required")
        if not text.strip():
            return []

        splitter = self._get_splitter()
        chunks: list[TextChunk] = []
        for index, (start_char, chunk_text) in enumerate(splitter.chunk_indices(text)):
            if not chunk_text.strip():
                continue
            chunks.append(
                TextChunk(
                    document_id=document_id,
                    chunk_id=f"{document_id}:{index:04d}",
                    text=chunk_text,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                )
            )
        return chunks
