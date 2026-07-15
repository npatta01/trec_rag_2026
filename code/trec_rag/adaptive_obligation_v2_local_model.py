"""Separately approval-gated, pinned local Qwen JSON generation adapter."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

from .adaptive_obligation_v2_contract import canonical_sha256
from .adaptive_obligation_v2_propose import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SNAPSHOT,
)


APPROVAL_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-approval-v1"
_FICLONE = 0x40049409
_HEX = frozenset("0123456789abcdef")


def verify_inference_approval(
    approval: object, preflight: Mapping[str, object]
) -> dict[str, object]:
    """Require the exact proposal-stage approval bound to this preflight."""

    if not isinstance(approval, Mapping) or not isinstance(preflight, Mapping):
        raise PermissionError("proposal inference approval required")
    receipt_sha256 = preflight.get("receipt_sha256")
    if (
        not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in receipt_sha256)
    ):
        raise PermissionError("proposal inference approval required")
    required = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "stage": "proposal",
        "preflight_sha256": receipt_sha256,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
        "approved": True,
    }
    if set(approval) != set(required) or any(
        approval.get(name) != value for name, value in required.items()
    ):
        raise PermissionError("proposal inference approval required")
    if (
        type(approval.get("primary_call_count")) is not int
        or type(approval.get("retry_call_ceiling")) is not int
        or type(approval.get("approved")) is not bool
        or type(preflight.get("primary_call_count")) is not int
        or type(preflight.get("retry_call_ceiling")) is not int
        or preflight.get("model") != MODEL_ID
        or preflight.get("model_revision") != MODEL_REVISION
        or preflight.get("primary_call_count") != 48
        or preflight.get("retry_call_ceiling") != 48
        or not isinstance(preflight.get("model_snapshot"), Mapping)
    ):
        raise PermissionError("proposal inference approval required")
    return dict(approval)


def _default_runtime_modules() -> object:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return SimpleNamespace(
        torch=torch,
        auto_tokenizer_cls=AutoTokenizer,
        auto_model_cls=AutoModelForCausalLM,
    )


class _PrivateSnapshot:
    """Own a private immutable-on-disk model view for one loaded runtime."""

    def __init__(self, temporary: tempfile.TemporaryDirectory[str], path: Path) -> None:
        self._temporary: tempfile.TemporaryDirectory[str] | None = temporary
        self.path = path

    def cleanup(self) -> None:
        if self._temporary is None:
            return
        try:
            if self.path.exists():
                os.chmod(self.path, 0o700)
                for child in self.path.iterdir():
                    os.chmod(child, 0o600, follow_symlinks=False)
        finally:
            temporary = self._temporary
            self._temporary = None
            if temporary is not None:
                temporary.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


def _validated_manifest(
    manifest: Mapping[str, object], *, model_id: str, revision: str
) -> list[dict[str, object]]:
    required_keys = {"model", "revision", "files", "manifest_sha256"}
    if set(manifest) != required_keys:
        raise RuntimeError("private model snapshot manifest keys differ")
    files = manifest.get("files")
    if (
        manifest.get("model") != model_id
        or manifest.get("revision") != revision
        or not isinstance(files, list)
    ):
        raise RuntimeError("private model snapshot manifest identity differs")
    rows: list[dict[str, object]] = []
    names: list[str] = []
    for value in files:
        if not isinstance(value, Mapping) or set(value) != {
            "name",
            "bytes",
            "blob_id",
            "content_sha256",
        }:
            raise RuntimeError("private model snapshot manifest row differs")
        name = value.get("name")
        byte_count = value.get("bytes")
        blob_id = value.get("blob_id")
        digest = value.get("content_sha256")
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or Path(name).name != name
            or type(byte_count) is not int
            or byte_count < 0
            or not isinstance(blob_id, str)
            or not blob_id
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in _HEX for char in digest)
        ):
            raise RuntimeError("private model snapshot manifest row is invalid")
        names.append(name)
        rows.append(dict(value))
    if len(names) != len(set(names)) or names != sorted(names):
        raise RuntimeError("private model snapshot manifest inventory differs")
    payload = {"model": model_id, "revision": revision, "files": rows}
    if manifest.get("manifest_sha256") != canonical_sha256(payload):
        raise RuntimeError("private model snapshot manifest hash differs")
    return rows


def _copy_or_reflink(source_fd: int, destination_fd: int) -> None:
    try:
        fcntl.ioctl(destination_fd, _FICLONE, source_fd)
        return
    except OSError:
        os.lseek(source_fd, 0, os.SEEK_SET)
        os.ftruncate(destination_fd, 0)
        os.lseek(destination_fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(source_fd, 1024 * 1024)
        if not chunk:
            break
        view = memoryview(chunk)
        while view:
            written = os.write(destination_fd, view)
            if written < 1:
                raise OSError("private model snapshot copy made no progress")
            view = view[written:]


def _sha256_fd(fd: int) -> tuple[int, str]:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


def _materialize_private_snapshot(
    snapshot: Path,
    manifest: Mapping[str, object],
    *,
    model_id: str,
    revision: str,
) -> _PrivateSnapshot:
    rows = _validated_manifest(manifest, model_id=model_id, revision=revision)
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.name != revision:
        raise RuntimeError("private model snapshot source is missing or unsafe")
    expected_names = [str(row["name"]) for row in rows]
    try:
        observed_names = sorted(child.name for child in snapshot.iterdir())
    except OSError as exc:
        raise RuntimeError("private model snapshot source is unavailable") from exc
    if observed_names != expected_names:
        raise RuntimeError("private model snapshot source inventory differs")

    temporary = tempfile.TemporaryDirectory(
        prefix="adaptive-obligation-v2-model-",
        dir="/var/tmp",
    )
    private = Path(temporary.name) / revision
    private.mkdir(mode=0o700)
    owner = _PrivateSnapshot(temporary, private)
    try:
        for row in rows:
            name = str(row["name"])
            source_fd = os.open(snapshot / name, os.O_RDONLY | os.O_CLOEXEC)
            try:
                if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                    raise RuntimeError(
                        "private model snapshot source entry is not regular"
                    )
                opened_blob_id = Path(
                    os.readlink(f"/proc/self/fd/{source_fd}")
                ).name
                if opened_blob_id != row["blob_id"]:
                    raise RuntimeError(
                        "private model snapshot source blob identity differs"
                    )
                destination_fd = os.open(
                    private / name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                )
                try:
                    _copy_or_reflink(source_fd, destination_fd)
                    os.fsync(destination_fd)
                finally:
                    os.close(destination_fd)
            finally:
                os.close(source_fd)

        if sorted(child.name for child in private.iterdir()) != expected_names:
            raise RuntimeError("private model snapshot inventory differs after copy")
        for row in rows:
            name = str(row["name"])
            fd = os.open(private / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                observed = os.fstat(fd)
                if not stat.S_ISREG(observed.st_mode):
                    raise RuntimeError("private model snapshot entry is not regular")
                size, digest = _sha256_fd(fd)
            finally:
                os.close(fd)
            if size != row["bytes"] or digest != row["content_sha256"]:
                raise RuntimeError("private model snapshot content hash differs")
            os.chmod(private / name, 0o400, follow_symlinks=False)
        os.chmod(private, 0o500)
        return owner
    except Exception as exc:
        owner.cleanup()
        if isinstance(exc, RuntimeError) and "private model snapshot" in str(exc):
            raise
        raise RuntimeError("private model snapshot materialization failed") from exc


class _PinnedLocalRuntime:
    def __init__(
        self,
        *,
        modules: object,
        tokenizer: object,
        model: object,
        private_snapshot: _PrivateSnapshot,
    ) -> None:
        self._modules = modules
        self._tokenizer = tokenizer
        self._model = model
        self._private_snapshot = private_snapshot

    def generate_json_bytes(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, object],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> tuple[bytes, int]:
        if not isinstance(schema, Mapping) or not schema:
            raise ValueError("proposal JSON schema is required")
        if do_sample is not False:
            raise ValueError("proposal generation must be deterministic")
        tokenizer = self._tokenizer
        encoded = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            [dict(row) for row in messages],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("local tokenizer returned invalid model inputs")
        device_inputs = {
            name: tensor.to("cuda")  # type: ignore[attr-defined]
            for name, tensor in encoded.items()
        }
        input_ids = device_inputs["input_ids"]
        prompt_tokens = int(input_ids.shape[-1])  # type: ignore[attr-defined]
        torch_module = self._modules.torch  # type: ignore[attr-defined]
        with torch_module.inference_mode():
            output = self._model.generate(  # type: ignore[attr-defined]
                **device_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = output[0][prompt_tokens:]
        output_token_count = int(generated.shape[-1])
        text = tokenizer.decode(generated, skip_special_tokens=True)  # type: ignore[attr-defined]
        if not isinstance(text, str):
            raise ValueError("local tokenizer decode did not return text")
        return text.encode("utf-8"), output_token_count


def load_pinned_local_runtime(
    *,
    model_id: str,
    revision: str,
    local_files_only: bool,
    model_manifest: Mapping[str, object],
    snapshot_dir: Path = MODEL_SNAPSHOT,
    runtime_modules: object | None = None,
) -> object:
    """Construct the exact local safetensor model on a verified ROCm device."""

    if (
        model_id != MODEL_ID
        or revision != MODEL_REVISION
        or local_files_only is not True
    ):
        raise RuntimeError("pinned local proposal model identity differs")
    private_snapshot = _materialize_private_snapshot(
        Path(snapshot_dir),
        model_manifest,
        model_id=model_id,
        revision=revision,
    )
    try:
        modules = (
            runtime_modules
            if runtime_modules is not None
            else _default_runtime_modules()
        )
        torch_module = modules.torch  # type: ignore[attr-defined]
        cuda = getattr(torch_module, "cuda", None)
        hip = getattr(getattr(torch_module, "version", None), "hip", None)
        if (
            not hip
            or cuda is None
            or not cuda.is_available()
            or int(cuda.device_count()) < 1
        ):
            raise RuntimeError("proposal inference requires an available ROCm device")
        tokenizer = modules.auto_tokenizer_cls.from_pretrained(  # type: ignore[attr-defined]
            private_snapshot.path,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = modules.auto_model_cls.from_pretrained(  # type: ignore[attr-defined]
            private_snapshot.path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch_module.bfloat16,
        )
        model = model.eval()  # type: ignore[attr-defined]
        model = model.to("cuda")  # type: ignore[attr-defined]
        return _PinnedLocalRuntime(
            modules=modules,
            tokenizer=tokenizer,
            model=model,
            private_snapshot=private_snapshot,
        )
    except Exception:
        private_snapshot.cleanup()
        raise


class V2LocalJsonModel:
    """Approval-gated facade over the pinned local JSON runtime."""

    def __init__(self, *, approval: object, preflight: Mapping[str, object]) -> None:
        verify_inference_approval(approval, preflight)
        manifest = preflight["model_snapshot"]
        if not isinstance(manifest, Mapping):
            raise PermissionError("proposal inference approval required")
        self._runtime = load_pinned_local_runtime(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            model_manifest=manifest,
        )

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, object],
        *,
        max_new_tokens: int,
    ) -> tuple[bytes, int]:
        return self._runtime.generate_json_bytes(  # type: ignore[attr-defined,no-any-return]
            messages,
            schema,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
