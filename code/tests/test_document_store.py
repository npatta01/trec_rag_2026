from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from threading import Barrier

import pytest

from trec_rag.document_store import (
    DocumentStore,
    DocumentStoreIntegrityError,
    ReadOnlyDocumentStore,
)


def test_document_store_preserves_exact_unicode_and_whitespace(tmp_path: Path) -> None:
    text = "Café\tline  one\nΩmega\n"
    store = DocumentStore(tmp_path / "objects")
    receipt = store.admit_text(text)
    assert receipt.content_sha256 == sha256(text.encode("utf-8")).hexdigest()
    assert receipt.byte_count == len(text.encode("utf-8"))
    assert receipt.character_count == len(text)
    assert store.read_text(receipt.content_sha256) == text


def test_document_store_rejects_expected_digest_mismatch(tmp_path: Path) -> None:
    with pytest.raises(DocumentStoreIntegrityError, match="expected"):
        DocumentStore(tmp_path).admit_text("body", expected_sha256="0" * 64)


def test_document_store_rejects_corrupt_existing_object(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path)
    receipt = store.admit_text("original")
    next(tmp_path.rglob("*.utf8")).write_bytes(b"changed")
    with pytest.raises(DocumentStoreIntegrityError, match="digest"):
        store.verify(receipt.content_sha256)


def test_document_store_identical_admission_is_idempotent(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path)
    assert store.admit_text("same") == store.admit_text("same")
    assert len(list(tmp_path.rglob("*.utf8"))) == 1


def test_document_store_simultaneous_identical_admission_is_idempotent(
    tmp_path: Path,
) -> None:
    store = DocumentStore(tmp_path)
    barrier = Barrier(2)

    def admit() -> object:
        barrier.wait(timeout=5)
        return store.admit_text("same concurrent body")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(admit) for _ in range(2)]
        receipts = [future.result(timeout=5) for future in futures]

    assert receipts[0] == receipts[1]
    assert len(list(tmp_path.rglob("*.utf8"))) == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_document_store_rejects_corrupt_object_on_admission_without_replacing_it(
    tmp_path: Path,
) -> None:
    text = "intended body"
    digest = sha256(text.encode("utf-8")).hexdigest()
    object_path = tmp_path / "sha256" / digest[:2] / f"{digest}.utf8"
    object_path.parent.mkdir(parents=True)
    corrupt_bytes = b"contradictory existing bytes"
    object_path.write_bytes(corrupt_bytes)

    with pytest.raises(DocumentStoreIntegrityError, match="digest"):
        DocumentStore(tmp_path).admit_text(text)

    assert object_path.read_bytes() == corrupt_bytes


def test_read_only_store_admits_only_existing_exact_object(tmp_path: Path) -> None:
    writable = DocumentStore(tmp_path)
    receipt = writable.admit_text("cached document")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    actual = ReadOnlyDocumentStore(tmp_path).admit_text(
        "cached document",
        expected_sha256=receipt.content_sha256,
    )

    assert actual == receipt
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_read_only_store_missing_object_creates_nothing(tmp_path: Path) -> None:
    root = tmp_path / "missing-store"

    with pytest.raises(DocumentStoreIntegrityError, match="unable to read"):
        ReadOnlyDocumentStore(root).admit_text("not cached")

    assert not root.exists()
