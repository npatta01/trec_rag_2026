from dataclasses import FrozenInstanceError
import json

import pytest

from trec_rag.tracing.models import (
    SpanSpec,
    TraceBundle,
    canonical_json_dumps,
    read_trace_bundle,
    write_trace_bundle,
)


def _bundle() -> TraceBundle:
    child = SpanSpec(
        name="child",
        kind="CHAIN",
        start_ns=1,
        end_ns=2,
        attributes={"trace.child": True},
        input_value={"documents": [{"docid": "d1", "text": "full text"}]},
        output_value={"answer": "final"},
        status="OK",
    )
    return TraceBundle(
        project_name="trace-project",
        session_id="session",
        topic_id="rag2026-1",
        baseline="ragnarok-fixed",
        root=SpanSpec(
            name="root",
            kind="CHAIN",
            start_ns=0,
            end_ns=3,
            attributes={"topic.id": "rag2026-1"},
            input_value=None,
            output_value=None,
            status="OK",
            children=(child,),
        ),
    )


def test_canonical_json_dumps_is_deterministic_strict_json():
    assert canonical_json_dumps({"z": "☃", "a": [2, 1]}) == '{"a":[2,1],"z":"☃"}'
    with pytest.raises(ValueError):
        canonical_json_dumps({"not_finite": float("nan")})


def test_trace_bundle_is_immutable_and_strict_json_round_trips_atomically(tmp_path):
    bundle = _bundle()
    path = tmp_path / "trace.json"

    assert write_trace_bundle(bundle, path) == path
    restored = read_trace_bundle(path)

    assert restored == bundle
    with pytest.raises(FrozenInstanceError):
        restored.session_id = "changed"
    with pytest.raises(TypeError):
        restored.root.attributes["topic.id"] = "changed"
    documents = restored.root.children[0].input_value["documents"]
    with pytest.raises(TypeError):
        documents.append({"rank": 2, "docid": "d2", "text": "other"})
    with pytest.raises(TypeError):
        documents[0]["text"] = "changed"

    original_bytes = path.read_bytes()
    path.write_text('{"project_name":"first","project_name":"second"}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_trace_bundle(path)
    path.write_bytes(original_bytes)

    child = bundle.root.children[0]
    object.__setattr__(child, "output_value", {"not_json": float("nan")})
    with pytest.raises(ValueError):
        write_trace_bundle(bundle, path)
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("project_name", 17),
        ("session_id", ["session"]),
        ("topic_id", {"topic": "rag2026-1"}),
        ("baseline", ["piika-agentic"]),
        ("root.name", 17),
        ("root.kind", ["CHAIN"]),
        ("root.status", 17),
        ("root.status_message", {"message": "bad"}),
    ],
    ids=[
        "project-name",
        "session-id",
        "topic-id",
        "baseline",
        "span-name",
        "span-kind",
        "span-status",
        "span-status-message",
    ],
)
def test_read_trace_bundle_rejects_non_string_contract_fields(
    tmp_path, field, invalid_value
):
    path = write_trace_bundle(_bundle(), tmp_path / "invalid-contract.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    if field.startswith("root."):
        record["root"][field.removeprefix("root.")] = invalid_value
    else:
        record[field] = invalid_value
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid (trace bundle|span value)"):
        read_trace_bundle(path)
