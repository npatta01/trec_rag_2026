from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import zipfile

import pytest

import trec_rag.experiments.organizer_pi.cli as trace_cli
from trec_rag.experiments.organizer_pi.cli import main
from trec_rag.tracing.phoenix_export import ExportReceipt, PhoenixSettings, SecretStr
from trec_rag.experiments.organizer_pi.event_trace import build_piika_trace
from trec_rag.tracing.models import read_trace_bundle, write_trace_bundle
from trec_rag.experiments.organizer_pi.inputs import OrganizerTopic


TOPIC = "rag2026-1"
NARRATIVE = "official narrative"
SECRET = "synthetic-phoenix-secret"


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def _write_events(path: Path, *, user_prompt: str | None = None) -> Path:
    rows = []
    if user_prompt is not None:
        user = {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
            "timestamp": 1_800_000_000_000,
        }
        rows.extend(
            [
                {"type": "message_start", "message": user},
                {"type": "message_end", "message": user},
            ]
        )
    assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": "answer"}],
        "stopReason": "stop",
        "timestamp": 1_800_000_000_100,
    }
    rows.extend(
        [
            {"type": "message_start", "message": assistant},
            {"type": "message_end", "message": assistant},
        ]
    )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _piika_record(path: Path, *, topic: str = TOPIC) -> Path:
    return _write_json(
        path,
        {
            "status": "completed",
            "query_id": topic,
            "trec_rag_output": {
                "metadata": {
                    "team_id": "team",
                    "narrative_id": topic,
                    "narrative": NARRATIVE,
                    "run_id": "run",
                    "run_desc": "description",
                },
                "references": ["d1"],
                "answer": [{"text": "answer", "citations": ["d1"]}],
            },
        },
    )


def _piika_args(tmp_path: Path) -> list[str]:
    topic_path = tmp_path / "topics.tsv"
    topic_path.write_text(f"{TOPIC}\t{NARRATIVE}\n", encoding="utf-8")
    return [
        "build",
        "--baseline",
        "piika-agentic",
        "--topic",
        TOPIC,
        "--topic-tsv",
        str(topic_path),
        "--events",
        str(_write_events(tmp_path / "events.jsonl")),
        "--record",
        str(_piika_record(tmp_path / "record.json")),
        "--bundle",
        str(tmp_path / "trace.json"),
    ]


def _fixed_script(path: Path) -> Path:
    path.write_text(
        "SYSTEM_PROMPT = 'exact system prompt'\n"
        "def prompt(question, docids, texts):\n"
        "    joined = '|'.join(f'{docid}:{texts[docid]}' for docid in docids)\n"
        "    return f'QUESTION={question}\\nDOCS={joined}'\n",
        encoding="utf-8",
    )
    return path


def _fixed_inputs(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    ranked_run = tmp_path / "top100.trec"
    ranked_run.write_text(
        "".join(
            f"{TOPIC} Q0 d{rank} {rank} {101 - rank}.0 fixed\n"
            for rank in range(1, 101)
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "documents.zip"
    first_text = " ".join(f"word-{index}" for index in range(1, 1002))
    with zipfile.ZipFile(archive, "w") as output:
        body = [
            json.dumps({"docid": "d1", "text": first_text}),
            *[
                json.dumps({"docid": f"d{rank}", "text": f"text {rank}"})
                for rank in range(2, 101)
            ],
            json.dumps({"docid": "not-ranked", "text": "must not be retained"}),
        ]
        output.writestr("documents.jsonl", "\n".join(body) + "\n")
    script = _fixed_script(tmp_path / "organizer.py")
    truncated = " ".join(first_text.split()[:1000])
    texts = {"d1": truncated, **{f"d{rank}": f"text {rank}" for rank in range(2, 101)}}
    rendered = "QUESTION=" + NARRATIVE + "\nDOCS=" + "|".join(
        f"d{rank}:{texts[f'd{rank}']}" for rank in range(1, 101)
    )
    return ranked_run, archive, script, rendered


def _fixed_record(path: Path, *, topic: str = TOPIC, narrative: str = NARRATIVE) -> Path:
    return _write_json(
        path,
        {
            "metadata": {
                "team_id": "team",
                "narrative_id": topic,
                "narrative": narrative,
                "run_id": "run",
                "run_desc": "description",
            },
            "references": ["d1"],
            "answer": [{"text": "answer", "citations": [0]}],
        },
    )


def _fixed_args(tmp_path: Path) -> list[str]:
    topic_path = tmp_path / "topics.tsv"
    topic_path.write_text(f"{TOPIC}\t{NARRATIVE}\n", encoding="utf-8")
    ranked_run, archive, script, rendered = _fixed_inputs(tmp_path)
    return [
        "build",
        "--baseline",
        "ragnarok-fixed",
        "--topic",
        TOPIC,
        "--topic-tsv",
        str(topic_path),
        "--events",
        str(_write_events(tmp_path / "fixed-events.jsonl", user_prompt=rendered)),
        "--record",
        str(_fixed_record(tmp_path / "fixed-record.json")),
        "--ranked-run",
        str(ranked_run),
        "--documents",
        str(archive),
        "--organizer-script",
        str(script),
        "--bundle",
        str(tmp_path / "fixed-trace.json"),
    ]


def _ignored(_path: Path) -> bool:
    return True


def _settings() -> PhoenixSettings:
    return PhoenixSettings(
        api_key=SecretStr(SECRET),
        collector_endpoint="https://phoenix.example.test",
        project_name="trec-rag-2026-pi-baselines",
    )


def _offline_export_kwargs():
    return {
        "settings_loader": _settings,
        "environment_loader": lambda: None,
        "credential_loader": lambda settings: (settings.api_key,),
        "ignored_checker": _ignored,
    }


def _bundle(path: Path):
    bundle = build_piika_trace(
        topic=OrganizerTopic(TOPIC, NARRATIVE),
        events=[
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                    "timestamp": 1_800_000_000_000,
                },
            }
        ],
        run_record={"status": "completed", "query_id": TOPIC},
        session_id=f"{TOPIC}-comparison",
    )
    write_trace_bundle(bundle, path)
    return bundle


def test_build_piika_writes_durable_bundle_without_loading_phoenix(tmp_path):
    def unexpected_settings():
        raise AssertionError("build must not load Phoenix settings")

    exit_code = main(
        _piika_args(tmp_path),
        ignored_checker=_ignored,
        settings_loader=unexpected_settings,
        environment_loader=lambda: pytest.fail("build must not load repository environment"),
    )

    assert exit_code == 0
    bundle = read_trace_bundle(tmp_path / "trace.json")
    assert bundle.baseline == "piika-agentic"
    assert bundle.topic_id == TOPIC
    assert bundle.session_id == f"{TOPIC}-comparison"
    assert bundle.root.children[-1].output_value["query_id"] == TOPIC


def test_fixed_build_streams_selected_documents_and_reconstructs_exact_prompts(tmp_path):
    exit_code = main(_fixed_args(tmp_path), ignored_checker=_ignored)

    assert exit_code == 0
    bundle = read_trace_bundle(tmp_path / "fixed-trace.json")
    documents = bundle.root.children[0].output_value["documents"]
    prompts = bundle.root.children[1].output_value
    assert len(documents) == 100
    assert [document["rank"] for document in documents] == list(range(1, 101))
    assert [document["docid"] for document in documents] == [
        f"d{rank}" for rank in range(1, 101)
    ]
    assert len(documents[0]["text"].split()) == 1000
    assert all(document["docid"] != "not-ranked" for document in documents)
    assert prompts["system_prompt"] == "exact system prompt"
    assert prompts["user_prompt"].startswith(f"QUESTION={NARRATIVE}\nDOCS=d1:word-1")
    assert bundle.root.children[-1].output_value == {
        "status": "completed",
        "trec_rag_output": json.loads((tmp_path / "fixed-record.json").read_text()),
    }


@pytest.mark.parametrize("missing", ["--ranked-run", "--documents", "--organizer-script"])
def test_fixed_build_requires_all_full_content_arguments_together(tmp_path, missing):
    args = _fixed_args(tmp_path)
    index = args.index(missing)
    del args[index : index + 2]

    assert main(args, ignored_checker=_ignored) == 2
    assert not (tmp_path / "fixed-trace.json").exists()


def test_build_rejects_missing_topic(tmp_path):
    args = _piika_args(tmp_path)
    args[args.index("--topic") + 1] = "rag2026-missing"

    assert main(args, ignored_checker=_ignored) == 1
    assert not (tmp_path / "trace.json").exists()


def test_build_rejects_output_record_for_a_different_topic(tmp_path):
    args = _piika_args(tmp_path)
    _piika_record(tmp_path / "record.json", topic="rag2026-other")

    assert main(args, ignored_checker=_ignored) == 1
    assert not (tmp_path / "trace.json").exists()


def test_fixed_build_rejects_output_narrative_mismatch(tmp_path):
    args = _fixed_args(tmp_path)
    _fixed_record(tmp_path / "fixed-record.json", narrative="different narrative")

    assert main(args, ignored_checker=_ignored) == 1
    assert not (tmp_path / "fixed-trace.json").exists()


def test_fixed_build_rejects_wrapped_record_for_a_different_topic(tmp_path):
    args = _fixed_args(tmp_path)
    successful_output = json.loads((tmp_path / "fixed-record.json").read_text())
    _write_json(
        tmp_path / "fixed-record.json",
        {
            "status": "completed",
            "query_id": "rag2026-other",
            "trec_rag_output": successful_output,
        },
    )

    assert main(args, ignored_checker=_ignored) == 1
    assert not (tmp_path / "fixed-trace.json").exists()


def test_fixed_build_retains_partial_failure_record_as_an_error_trace(tmp_path):
    args = _fixed_args(tmp_path)
    _, _, _, rendered = _fixed_inputs(tmp_path)
    _write_json(
        tmp_path / "fixed-record.json",
        {"status": "failed", "query_id": TOPIC, "error": "validation failed"},
    )
    (tmp_path / "fixed-events.jsonl").write_text(
        json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": rendered}],
                    "timestamp": 1_800_000_000_000,
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "extension_error",
                "timestamp": 1_800_000_000_001,
                "error": "generation failed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(args, ignored_checker=_ignored) == 0
    bundle = read_trace_bundle(tmp_path / "fixed-trace.json")
    assert bundle.root.status == "ERROR"
    assert bundle.root.children[2].status == "ERROR"
    assert bundle.root.children[-1].status == "ERROR"
    assert bundle.root.children[-1].output_value == {
        "status": "failed",
        "query_id": TOPIC,
        "error": "validation failed",
    }


def test_fixed_build_rejects_native_user_prompt_mismatch(tmp_path):
    args = _fixed_args(tmp_path)
    _write_events(tmp_path / "fixed-events.jsonl", user_prompt="modified prompt")

    assert main(args, ignored_checker=_ignored) == 1
    assert not (tmp_path / "fixed-trace.json").exists()


def test_fixed_build_accepts_pi_file_envelope_around_exact_native_prompt(tmp_path):
    args = _fixed_args(tmp_path)
    _, _, _, rendered = _fixed_inputs(tmp_path)
    wrapped = f'<file name="/tmp/ragnarok-random/prompt.txt">\n{rendered}\n</file>\n'
    _write_events(tmp_path / "fixed-events.jsonl", user_prompt=wrapped)

    assert main(args, ignored_checker=_ignored) == 0
    assert (tmp_path / "fixed-trace.json").is_file()


def test_fixed_build_rejects_events_without_a_native_user_prompt(tmp_path):
    args = _fixed_args(tmp_path)
    _write_events(tmp_path / "fixed-events.jsonl")

    assert main(args, ignored_checker=_ignored) == 1
    assert not (tmp_path / "fixed-trace.json").exists()


def test_fixed_build_rejects_an_extra_unrelated_native_user_prompt(tmp_path):
    args = _fixed_args(tmp_path)
    _, _, _, rendered = _fixed_inputs(tmp_path)
    event_path = _write_events(tmp_path / "fixed-events.jsonl", user_prompt=rendered)
    rows = [json.loads(line) for line in event_path.read_text().splitlines()]
    extra = {
        "type": "message_end",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "unrelated extra prompt"}],
            "timestamp": 1_800_000_000_050,
        },
    }
    rows.insert(2, extra)
    event_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert main(args, ignored_checker=_ignored) == 1
    assert not (tmp_path / "fixed-trace.json").exists()


def test_fixed_build_rejects_malformed_pi_file_envelope_attributes(tmp_path):
    args = _fixed_args(tmp_path)
    _, _, _, rendered = _fixed_inputs(tmp_path)
    malformed = (
        f'<file name="/tmp/ragnarok-random/prompt.txt" mode="trusted">\n'
        f"{rendered}\n</file>\n"
    )
    _write_events(tmp_path / "fixed-events.jsonl", user_prompt=malformed)

    assert main(args, ignored_checker=_ignored) == 1
    assert not (tmp_path / "fixed-trace.json").exists()


def test_build_rejects_non_ignored_bundle_target(tmp_path):
    assert main(_piika_args(tmp_path), ignored_checker=lambda _path: False) == 1
    assert not (tmp_path / "trace.json").exists()


def test_export_writes_only_public_receipt_fields_after_success(tmp_path):
    bundle_path = tmp_path / "trace.json"
    receipt_path = tmp_path / "receipt.json"
    expected = ExportReceipt(
        project_name="trec-rag-2026-pi-baselines",
        trace_id="abc123",
        root_span_id="def456",
        exported_span_count=4,
    )

    exit_code = main(
        ["export", "--bundle", str(bundle_path), "--receipt", str(receipt_path)],
        bundle_loader=lambda path: _bundle(path),
        exporter=lambda bundle, settings: expected,
        **_offline_export_kwargs(),
    )

    assert exit_code == 0
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == asdict(expected)
    assert SECRET not in receipt_path.read_text(encoding="utf-8")


def test_export_failure_preserves_bundle_and_does_not_write_receipt(tmp_path, capsys):
    bundle_path = tmp_path / "trace.json"
    receipt_path = tmp_path / "receipt.json"
    _bundle(bundle_path)
    original = bundle_path.read_bytes()
    saw_staged_receipt = False

    def fail_export(_bundle, _settings):
        nonlocal saw_staged_receipt
        saw_staged_receipt = len(list(tmp_path.glob(".receipt.json.*"))) == 1
        raise RuntimeError(f"collector rejected {SECRET}")

    exit_code = main(
        ["export", "--bundle", str(bundle_path), "--receipt", str(receipt_path)],
        exporter=fail_export,
        **_offline_export_kwargs(),
    )

    assert exit_code == 1
    assert saw_staged_receipt is True
    assert bundle_path.read_bytes() == original
    assert not receipt_path.exists()
    assert list(tmp_path.glob(".receipt.json.*")) == []
    assert SECRET not in capsys.readouterr().err


def test_receipt_staging_failure_prevents_export(tmp_path, monkeypatch):
    bundle_path = tmp_path / "trace.json"
    receipt_path = tmp_path / "receipt.json"
    _bundle(bundle_path)
    called = False

    def fail_staging(*_args, **_kwargs):
        raise OSError("synthetic receipt staging failure")

    def exporter(_bundle, _settings):
        nonlocal called
        called = True
        raise AssertionError("export must start only after receipt staging")

    monkeypatch.setattr(trace_cli.tempfile, "mkstemp", fail_staging)

    exit_code = main(
        ["export", "--bundle", str(bundle_path), "--receipt", str(receipt_path)],
        exporter=exporter,
        **_offline_export_kwargs(),
    )

    assert exit_code == 1
    assert called is False
    assert not receipt_path.exists()
    assert list(tmp_path.glob(".receipt.json.*")) == []


def test_receipt_stream_setup_failure_closes_descriptor_and_removes_temp(
    tmp_path, monkeypatch
):
    bundle_path = tmp_path / "trace.json"
    receipt_path = tmp_path / "receipt.json"
    _bundle(bundle_path)
    real_mkstemp = trace_cli.tempfile.mkstemp
    allocated: list[tuple[int, Path]] = []
    called = False

    def capture_staging(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        allocated.append((descriptor, Path(name)))
        return descriptor, name

    def fail_stream_setup(*_args, **_kwargs):
        raise OSError("synthetic fdopen failure")

    def exporter(_bundle, _settings):
        nonlocal called
        called = True
        raise AssertionError("export must start only after receipt stream setup")

    monkeypatch.setattr(trace_cli.tempfile, "mkstemp", capture_staging)
    monkeypatch.setattr(trace_cli.os, "fdopen", fail_stream_setup)

    exit_code = main(
        ["export", "--bundle", str(bundle_path), "--receipt", str(receipt_path)],
        exporter=exporter,
        **_offline_export_kwargs(),
    )

    assert exit_code == 1
    assert called is False
    assert len(allocated) == 1
    descriptor, temporary_path = allocated[0]
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert not temporary_path.exists()
    assert not receipt_path.exists()


def test_export_scans_bundle_for_credentials_before_calling_exporter(tmp_path, capsys):
    bundle_path = tmp_path / "trace.json"
    receipt_path = tmp_path / "receipt.json"
    bundle = _bundle(bundle_path)
    contaminated = build_piika_trace(
        topic=OrganizerTopic(TOPIC, NARRATIVE),
        events=[
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"leak {SECRET}"}],
                    "timestamp": 1_800_000_000_000,
                },
            }
        ],
        run_record=bundle.root.children[-1].output_value,
        session_id=bundle.session_id,
    )
    write_trace_bundle(contaminated, bundle_path)
    called = False

    def exporter(_bundle, _settings):
        nonlocal called
        called = True
        raise AssertionError("secret scan must run first")

    exit_code = main(
        ["export", "--bundle", str(bundle_path), "--receipt", str(receipt_path)],
        exporter=exporter,
        **_offline_export_kwargs(),
    )

    assert exit_code == 1
    assert called is False
    assert not receipt_path.exists()
    assert SECRET not in capsys.readouterr().err


def test_export_rejects_non_ignored_receipt_target(tmp_path):
    bundle_path = tmp_path / "trace.json"
    receipt_path = tmp_path / "receipt.json"
    _bundle(bundle_path)
    checked = []

    def only_bundle_is_ignored(path):
        checked.append(path)
        return path == bundle_path

    assert main(
        ["export", "--bundle", str(bundle_path), "--receipt", str(receipt_path)],
        exporter=lambda _bundle, _settings: pytest.fail("must not export"),
        **{
            **_offline_export_kwargs(),
            "ignored_checker": only_bundle_is_ignored,
        },
    ) == 1
    assert checked == [bundle_path, receipt_path]
    assert not receipt_path.exists()
