from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trec_rag.adaptive_evidence_contract import (
    _load_cli_sources,
    build_contract,
    canonical_sha256,
    document_fold,
    main,
    reject_protected_before_access,
    render_obligation_query,
)


def test_canonical_sha256_uses_sorted_compact_utf8_json() -> None:
    value = {"z": "café", "a": [2, 1]}
    expected = hashlib.sha256(b'{"a":[2,1],"z":"caf\xc3\xa9"}').hexdigest()
    assert canonical_sha256(value) == expected


def _manifest() -> dict[str, object]:
    return {
        "topics": [{"topic_id": "219", "query": "full narrative"}],
        "facets": [{
            "topic_id": "219",
            "facet_id": "219-positive",
            "manifest_order": 0,
            "obligation": "Positive effects on society.",
            "anchor_terms": ["technology"],
            "relation_terms": ["positive", "society"],
            "wrong_domain_patterns": ["technology stock"],
        }],
    }


def _rows() -> list[dict[str, object]]:
    return [{
        "topic_id": "219",
        "document_id": "d1",
        "text": "technology can affect society",
        "text_sha256": hashlib.sha256(b"technology can affect society").hexdigest(),
        "union_order": 1,
        "provenance": [{"family": "facet", "facet_id": "219-positive", "rank": 1}],
    }]


def _gates() -> dict[str, object]:
    return {"gates": [{"facet_id": "219-positive", "accepted": True}]}


def test_contract_tethers_o0_to_full_narrative_and_assigns_fold() -> None:
    contract = build_contract(
        _manifest(), _gates(), _rows(), expected_population=1, expected_o0=1,
    )
    o0 = contract["obligations"][1]
    assert contract["obligations"][0]["kind"] == "broad"
    assert o0["kind"] == "o0"
    assert o0["query"] == "full narrative\n\nExplicit obligation:\nPositive effects on society."
    assert contract["documents"][0]["fold"] == document_fold("219", "d1")


def test_protected_topic_fails_before_loader() -> None:
    touched = False
    def loader() -> object:
        nonlocal touched
        touched = True
        return object()
    with pytest.raises(ValueError, match="protected topic 144"):
        reject_protected_before_access(["144"], loader)
    assert touched is False


def test_fold_is_sha256_mod_two() -> None:
    expected = int(hashlib.sha256(b"219\0d1").hexdigest(), 16) % 2
    assert document_fold("219", "d1") == expected


def test_contract_rejects_a_protected_accepted_facet() -> None:
    manifest = _manifest()
    manifest["facets"][0]["topic_id"] = "144"  # type: ignore[index]
    with pytest.raises(ValueError, match="protected topic 144"):
        build_contract(manifest, _gates(), _rows(), expected_population=1, expected_o0=1)


def test_contract_preflights_protected_facet_metadata_before_gate_access() -> None:
    accesses: list[str] = []

    class RecordingGates(dict[str, object]):
        def __getitem__(self, key: str) -> object:
            accesses.append(f"gates:item:{key}")
            return super().__getitem__(key)

    manifest = _manifest()
    manifest["facets"][0]["topic_id"] = "144"  # type: ignore[index]
    with pytest.raises(ValueError, match="protected topic 144"):
        build_contract(
            manifest,
            RecordingGates(_gates()),
            _rows(),
            expected_population=1,
            expected_o0=1,
        )

    assert accesses == []


def test_contract_preflights_protected_metadata_before_gate_union_or_join_access() -> None:
    accesses: list[str] = []

    class RecordingDict(dict[str, object]):
        def __init__(self, label: str, value: dict[str, object]) -> None:
            super().__init__(value)
            self.label = label

        def get(self, key: str, default: object = None) -> object:
            accesses.append(f"{self.label}:get:{key}")
            return super().get(key, default)

        def __getitem__(self, key: str) -> object:
            accesses.append(f"{self.label}:item:{key}")
            return super().__getitem__(key)

    class RecordingRows(list[dict[str, object]]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            accesses.append("union:iter")
            return super().__iter__()

    manifest = RecordingDict(
        "manifest",
        {
            "topic_ids": ["144"],
            "topics": [{"topic_id": "144", "query": "forbidden"}],
            "facets": [],
        },
    )
    gates = RecordingDict("gates", {"gates": []})

    with pytest.raises(ValueError, match="protected topic 144"):
        build_contract(
            manifest,
            gates,
            RecordingRows(),
            expected_population=0,
            expected_o0=0,
        )

    assert accesses == ["manifest:get:topic_ids"]


def test_contract_preflights_protected_gate_before_obligation_or_union_access() -> None:
    accesses: list[str] = []

    class RecordingTopic(dict[str, object]):
        def __getitem__(self, key: str) -> object:
            if key == "query":
                accesses.append("obligation:query")
            return super().__getitem__(key)

    class RecordingRows(list[dict[str, object]]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            accesses.append("union:iter")
            return super().__iter__()

    manifest = _manifest()
    manifest["topics"][0] = RecordingTopic(  # type: ignore[index]
        {"topic_id": "219", "query": "full narrative"}
    )
    gates = {
        "gates": [
            {
                "topic_id": "144",
                "facet_id": "219-positive",
                "accepted": True,
            }
        ]
    }

    with pytest.raises(ValueError, match="protected topic 144"):
        build_contract(
            manifest,
            gates,
            RecordingRows(),
            expected_population=0,
            expected_o0=1,
        )

    assert accesses == []


def test_contract_rejects_accepted_gate_id_missing_from_manifest() -> None:
    gates = {"gates": [{"facet_id": "219-missing", "accepted": True}]}
    with pytest.raises(ValueError, match="accepted facet set.*manifest"):
        build_contract(_manifest(), gates, _rows(), expected_population=1, expected_o0=1)


def test_contract_rejects_union_topic_outside_manifest() -> None:
    rows = _rows()
    rows[0]["topic_id"] = "999"
    with pytest.raises(ValueError, match="outside the manifest topic set"):
        build_contract(_manifest(), _gates(), rows, expected_population=1, expected_o0=1)


def _write_cli_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    topic_ids = ["219", "72", "300", "84"]
    topics = [
        {"topic_id": topic_id, "query": f"narrative {topic_id}"}
        for topic_id in topic_ids
    ]
    facets = []
    gates = []
    for order in range(24):
        topic_id = topic_ids[order % len(topic_ids)]
        facet_id = f"{topic_id}-facet-{order}"
        facets.append(
            {
                "topic_id": topic_id,
                "facet_id": facet_id,
                "manifest_order": order,
                "obligation": f"Obligation {order}.",
                "anchor_terms": [f"anchor-{order}"],
                "relation_terms": [f"relation-{order}"],
                "wrong_domain_patterns": [f"wrong-{order}"],
            }
        )
        gates.append({"topic_id": topic_id, "facet_id": facet_id, "accepted": True})

    manifest_path = tmp_path / "source-manifest.json"
    gate_path = tmp_path / "gates.json"
    union_path = tmp_path / "u_accepted.jsonl"
    manifest_path.write_text(
        json.dumps({"topic_ids": topic_ids, "topics": topics, "facets": facets}),
        encoding="utf-8",
    )
    gate_path.write_text(json.dumps({"gates": gates}), encoding="utf-8")
    with union_path.open("w", encoding="utf-8") as handle:
        for order in range(8114):
            topic_id = topic_ids[order % len(topic_ids)]
            text = f"evidence {order}"
            row = {
                "topic_id": topic_id,
                "document_id": f"d{order}",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "union_order": order + 1,
                "provenance": [{"family": "original", "rank": order + 1}],
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return manifest_path, gate_path, union_path


def test_create_cli_writes_canonical_hashed_contract_and_rejects_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path, gate_path, union_path = _write_cli_inputs(tmp_path)
    output = tmp_path / "contract"

    assert main(
        [
            "create",
            "--manifest", str(manifest_path),
            "--gates", str(gate_path),
            "--union", str(union_path),
            "--output", str(output),
        ]
    ) == 0
    assert capsys.readouterr().out.strip() == (
        "status=complete documents=8114 o0=24 protected=0 qrels_opened=false"
    )
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "obligations.jsonl",
        "documents.jsonl",
        "folds.jsonl",
        "summary.json",
    }

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["document_count"] == 8114
    assert summary["o0_obligation_count"] == 24
    assert summary["qrels_opened"] is False
    assert set(summary["artifact_sha256"]) == {
        "manifest.json", "obligations.jsonl", "documents.jsonl", "folds.jsonl",
    }
    for name, digest in summary["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert len((output / "obligations.jsonl").read_text().splitlines()) == 28
    assert len((output / "documents.jsonl").read_text().splitlines()) == 8114
    assert len((output / "folds.jsonl").read_text().splitlines()) == 8114

    manifest_path.unlink()
    gate_path.unlink()
    union_path.unlink()
    with pytest.raises(FileExistsError):
        main(
            [
                "create",
                "--manifest", str(manifest_path),
                "--gates", str(gate_path),
                "--union", str(union_path),
                "--output", str(output),
            ]
        )


def test_create_cli_rejects_protected_manifest_before_other_source_access(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "protected.json"
    manifest_path.write_text(
        json.dumps(
            {
                "topic_ids": ["144", "72", "300", "84"],
                "topics": [{"topic_id": "144", "query": "forbidden"}],
                "facets": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="protected topic 144"):
        main(
            [
                "create",
                "--manifest", str(manifest_path),
                "--gates", str(tmp_path / "missing-gates.json"),
                "--union", str(tmp_path / "missing-union.jsonl"),
                "--output", str(tmp_path / "contract"),
            ]
        )


def test_cli_protected_metadata_prevents_gate_union_and_join_loader_calls() -> None:
    accesses: list[str] = []

    def record(label: str, result: object):
        def loader() -> object:
            accesses.append(label)
            return result

        return loader

    with pytest.raises(ValueError, match="protected topic 144"):
        _load_cli_sources(
            manifest_topic_ids_loader=record(
                "manifest_metadata",
                (
                    ["144", "72", "300", "84"],
                    ["144", "72", "300", "84"],
                ),
            ),
            manifest_loader=record("manifest_join", ({}, b"")),
            gates_loader=record("gates", ({}, b"")),
            union_loader=record("union", ([], b"")),
        )

    assert accesses == ["manifest_metadata"]


def test_create_cli_rejects_any_declared_nonpilot_topic_set_before_gate_access(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "wrong-topic-set.json"
    manifest_path.write_text(
        json.dumps(
            {
                "topic_ids": ["219", "72", "300", "999"],
                "topics": [
                    {"topic_id": topic_id, "query": f"narrative {topic_id}"}
                    for topic_id in ("219", "72", "300", "84")
                ],
                "facets": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly the four pilot topics"):
        main(
            [
                "create",
                "--manifest", str(manifest_path),
                "--gates", str(tmp_path / "missing-gates.json"),
                "--union", str(tmp_path / "missing-union.jsonl"),
                "--output", str(tmp_path / "contract"),
            ]
        )
