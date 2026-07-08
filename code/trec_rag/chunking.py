"""Stable chunking contracts for reranking and evidence selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class MissingChunkerDependency(RuntimeError):
    """Raised when an optional chunking backend is not installed."""


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


def _import_text_splitter():
    try:
        from semantic_text_splitter import TextSplitter
    except ImportError as exc:
        raise MissingChunkerDependency(
            "semantic-text-splitter is required for SemanticTextChunker. "
            "Run with: uv run --with semantic-text-splitter ..."
        ) from exc
    return TextSplitter


class SemanticTextChunker:
    """Text chunker backed by semantic-text-splitter.

    Keep callers on the TextChunker contract so the backend can later change
    without changing reranking or evidence-selection code.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def split_text(self, text: str, *, document_id: str) -> list[TextChunk]:
        if not document_id.strip():
            raise ValueError("document_id is required")
        if not text.strip():
            return []

        splitter = _import_text_splitter()(
            self.config.max_characters,
            overlap=self.config.overlap_characters,
            trim=self.config.trim,
        )
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
