import pytest

from trec_rag.chunking import ChunkingConfig, MissingChunkerDependency, SemanticTextChunker


def test_semantic_text_chunker_preserves_stable_chunk_contract():
    text = (
        "Title\n\n"
        "Alpha section explains bank failures and regulation. "
        "It has enough words to require chunking. "
        "Beta section compares banks and credit unions. "
        "Gamma section discusses investor risk and deposit insurance. "
    ) * 4
    chunker = SemanticTextChunker(
        ChunkingConfig(max_characters=180, overlap_characters=30)
    )

    chunks = chunker.split_text(text, document_id="doc-a")

    assert len(chunks) > 1
    assert [chunk.chunk_id for chunk in chunks] == [
        f"doc-a:{index:04d}" for index in range(len(chunks))
    ]
    assert all(chunk.document_id == "doc-a" for chunk in chunks)
    assert all(0 <= chunk.start_char < chunk.end_char <= len(text) for chunk in chunks)
    assert all(text[chunk.start_char : chunk.end_char] == chunk.text for chunk in chunks)
    assert all(len(chunk.text) <= 180 for chunk in chunks)


def test_semantic_text_chunker_rejects_invalid_config():
    with pytest.raises(ValueError, match="max_characters"):
        ChunkingConfig(max_characters=0)

    with pytest.raises(ValueError, match="overlap_characters"):
        ChunkingConfig(max_characters=100, overlap_characters=100)


def test_semantic_text_chunker_trims_empty_text_to_no_chunks():
    chunker = SemanticTextChunker(ChunkingConfig(max_characters=100, overlap_characters=10))

    assert chunker.split_text(" \n\n ", document_id="doc-empty") == []


def test_missing_semantic_text_splitter_dependency_has_actionable_error(monkeypatch):
    import trec_rag.chunking as chunking

    real_import = chunking._import_text_splitter

    def missing_import():
        raise MissingChunkerDependency("install semantic-text-splitter")

    monkeypatch.setattr(chunking, "_import_text_splitter", missing_import)
    chunker = SemanticTextChunker(ChunkingConfig(max_characters=100, overlap_characters=10))

    with pytest.raises(MissingChunkerDependency, match="semantic-text-splitter"):
        chunker.split_text("content", document_id="doc-a")

    monkeypatch.setattr(chunking, "_import_text_splitter", real_import)
