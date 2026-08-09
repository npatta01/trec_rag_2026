"""Exact UTF-8 content-addressed document storage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import tempfile


DOCUMENT_STORE_SCHEMA_VERSION = "document-store-v1"
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


class DocumentStoreIntegrityError(RuntimeError):
    """Raised when a document object does not match its content address."""


@dataclass(frozen=True)
class DocumentReceipt:
    content_sha256: str
    byte_count: int
    character_count: int


class DocumentStore:
    def __init__(self, root: Path):
        self._root = Path(root)

    def admit_text(
        self, text: str, *, expected_sha256: str | None = None
    ) -> DocumentReceipt:
        if expected_sha256 is not None:
            self._validate_digest(expected_sha256)

        try:
            content = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise DocumentStoreIntegrityError("text is not valid UTF-8") from exc

        receipt = DocumentReceipt(
            content_sha256=hashlib.sha256(content).hexdigest(),
            byte_count=len(content),
            character_count=len(text),
        )
        if expected_sha256 is not None and expected_sha256 != receipt.content_sha256:
            raise DocumentStoreIntegrityError(
                "expected digest does not match admitted text: "
                f"expected {expected_sha256}, got {receipt.content_sha256}"
            )

        destination = self._object_path(receipt.content_sha256)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(name)
            with os.fdopen(descriptor, "wb") as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                return self.verify(receipt.content_sha256)
            self._fsync_directory(destination.parent)
            return receipt
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def read_text(self, digest: str) -> str:
        content, _ = self._read_verified(digest)
        return content.decode("utf-8")

    def verify(self, digest: str) -> DocumentReceipt:
        _, receipt = self._read_verified(digest)
        return receipt

    def _read_verified(self, digest: str) -> tuple[bytes, DocumentReceipt]:
        self._validate_digest(digest)
        path = self._object_path(digest)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise DocumentStoreIntegrityError(
                f"unable to read document object {digest}"
            ) from exc
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentStoreIntegrityError(
                f"document object {digest} is not valid UTF-8"
            ) from exc
        receipt = DocumentReceipt(
            content_sha256=hashlib.sha256(content).hexdigest(),
            byte_count=len(content),
            character_count=len(text),
        )
        if receipt.content_sha256 != digest:
            raise DocumentStoreIntegrityError(
                f"document digest mismatch: expected {digest}, got {receipt.content_sha256}"
            )
        return content, receipt

    def _object_path(self, digest: str) -> Path:
        self._validate_digest(digest)
        return self._root / "sha256" / digest[:2] / f"{digest}.utf8"

    @staticmethod
    def _validate_digest(digest: str) -> None:
        if not isinstance(digest, str) or _DIGEST_PATTERN.fullmatch(digest) is None:
            raise DocumentStoreIntegrityError(
                "digest must be a lowercase 64-hex SHA-256 value"
            )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class ReadOnlyDocumentStore(DocumentStore):
    """Document store that can verify existing objects but never admit new ones."""

    def admit_text(
        self, text: str, *, expected_sha256: str | None = None
    ) -> DocumentReceipt:
        if expected_sha256 is not None:
            self._validate_digest(expected_sha256)

        try:
            content = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise DocumentStoreIntegrityError("text is not valid UTF-8") from exc

        receipt = DocumentReceipt(
            content_sha256=hashlib.sha256(content).hexdigest(),
            byte_count=len(content),
            character_count=len(text),
        )
        if expected_sha256 is not None and expected_sha256 != receipt.content_sha256:
            raise DocumentStoreIntegrityError(
                "expected digest does not match admitted text: "
                f"expected {expected_sha256}, got {receipt.content_sha256}"
            )

        stored_content, stored_receipt = self._read_verified(receipt.content_sha256)
        if stored_content != content:
            raise DocumentStoreIntegrityError(
                "cached document object does not match admitted text"
            )
        return stored_receipt
