"""A generic synthetic run fixture for offline-evaluation tests.

It extends the debug-report fixture builders instead of restating their sealed schemas,
and takes arbitrary topic identifiers so nothing in the tests depends on the shape or the
names of any real competition run.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

from test_competition_debug_report import (  # noqa: E402  (sibling test module)
    _file_receipt,
    _json_bytes,
    _rag_generation_topic,
    _sha256,
    _write_debug_handoff,
    _write_debug_run,
)

BASE_TOPIC = "rag2026-0"
BASE_NARRATIVE = 'Original <narrative> & "quotes"'
RUN_DESC = "Fixture answer generation."


@dataclass(frozen=True)
class TopicSpec:
    """One synthetic topic: arbitrary id, narrative, candidate depth, and answer shape."""

    topic_id: str
    narrative: str
    candidate_documents: int = 5
    answers: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("First supported answer.", (0,)),
        ("Second detail.", (0,)),
    )
    # The evidence passage the judge actually sees. Two runs that share this and their
    # statements produce byte-equivalent judge requests, which is what cross-run cache
    # reuse depends on.
    evidence_text: str = "Shared selected evidence passage."


@dataclass
class RunFixture:
    root: Path
    retrieval_config: Path
    rag_config: Path
    retrieval_output: Path
    rag_output: Path
    work_dir: Path
    specs: tuple[TopicSpec, ...]
    experiment_id: str
    run_id: str
    docids: dict[str, str] = field(default_factory=dict)
    accepted_bundle_metadata: Path | None = None
    accepted_handoff: Path | None = None

    @property
    def topic_ids(self) -> tuple[str, ...]:
        return tuple(spec.topic_id for spec in self.specs)

    def narratives(self) -> dict[str, str]:
        return {spec.topic_id: spec.narrative for spec in self.specs}

    @property
    def bundle_metadata(self) -> Path:
        if self.accepted_bundle_metadata is None:
            raise AttributeError("this fixture has no accepted bundle metadata")
        return self.accepted_bundle_metadata

    @property
    def handoff(self) -> Path:
        if self.accepted_handoff is None:
            raise AttributeError("this fixture has no accepted handoff")
        return self.accepted_handoff

    def qrels(self, topics: Sequence[str] | None = None, *, grade: int = 1) -> Path:
        """Write qrels naming this run's real submitted docid for the given topics."""
        selected = list(self.topic_ids if topics is None else topics)
        path = self.root / f"qrels-{'-'.join(selected) or 'empty'}.txt"
        lines = [f"{topic_id} 0 {self.docids[topic_id]} {grade}" for topic_id in selected]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def gold_nuggets(self, topics: Sequence[str] | None = None) -> Path:
        selected = list(self.topic_ids if topics is None else topics)
        path = self.root / f"gold-{'-'.join(selected) or 'empty'}.jsonl"
        rows = [
            {
                "qid": topic_id,
                "query": self.narratives().get(topic_id, topic_id),
                "nuggets": [{"text": f"gold nugget for {topic_id}", "importance": "vital"}],
            }
            for topic_id in selected
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path


def rewrite(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [rewrite(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item, mapping) for key, item in value.items()}
    return value


def _rewrite_tree(root: Path, mapping: dict[str, str]) -> None:
    for path in root.rglob("*.json"):
        path.write_bytes(_json_bytes(rewrite(json.loads(path.read_text(encoding="utf-8")), mapping)))
    for path in root.rglob("*.jsonl"):
        rows = [
            rewrite(json.loads(line), mapping)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        path.write_bytes(b"".join(_json_bytes(row) for row in rows))


def _reseal_topic(topic_root: Path) -> tuple[str, str]:
    """Re-seal one topic tree after a rewrite; return its scoring and canonical digests."""
    scoring_manifest_path = topic_root / "scoring" / "complete.json"
    scoring_manifest = json.loads(scoring_manifest_path.read_text(encoding="utf-8"))
    scoring_manifest["artifacts"][0] = {
        "relative_path": "scoring/lane_scores.jsonl",
        **_file_receipt(topic_root / "scoring" / "lane_scores.jsonl"),
    }
    scoring_manifest_path.write_bytes(_json_bytes(scoring_manifest))
    scoring_digest = str(_file_receipt(scoring_manifest_path)["sha256"])

    canonical = topic_root / "canonical"
    nugget_manifest_path = canonical / "canonical-nugget-manifest.json"
    nugget_manifest = json.loads(nugget_manifest_path.read_text(encoding="utf-8"))
    nugget_manifest["selections_sha256"] = sha256(
        (canonical / "subnarrative-selections.jsonl").read_bytes()
    ).hexdigest()
    nugget_manifest["selection_manifest_sha256"] = sha256(
        (canonical / "selection-manifest.json").read_bytes()
    ).hexdigest()
    nugget_manifest["canonical_nuggets_sha256"] = sha256(
        (canonical / "canonical-nuggets.jsonl").read_bytes()
    ).hexdigest()
    nugget_manifest_path.write_bytes(_json_bytes(nugget_manifest))

    checkpoint_path = canonical / "complete.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    for artifact in checkpoint["artifacts"]:
        artifact.update(_file_receipt(topic_root / artifact["relative_path"]))
    checkpoint_path.write_bytes(_json_bytes(checkpoint))
    return scoring_digest, sha256(checkpoint_path.read_bytes()).hexdigest()


def build_run(
    root: Path,
    specs: Sequence[TopicSpec],
    *,
    depth_kind: str = "natural_union",
    experiment_id: str = "debug-fixture",
    shuffle_rows: bool = False,
) -> RunFixture:
    """Materialize a complete sealed retrieval export plus a completed RAG run."""
    if not specs:
        raise ValueError("at least one topic spec is required")
    retrieval_config, output = _write_debug_run(root)
    if experiment_id != "debug-fixture":
        retrieval_config.write_text(
            retrieval_config.read_text(encoding="utf-8").replace(
                "id: debug-fixture", f"id: {experiment_id}"
            ),
            encoding="utf-8",
        )
        renamed = output.with_name(experiment_id)
        output.rename(renamed)
        output = renamed

    topics_path = root / "topics.jsonl"
    topics_path.write_text(
        "".join(
            json.dumps({"id": spec.topic_id, "narrative": spec.narrative}) + "\n"
            for spec in specs
        ),
        encoding="utf-8",
    )

    base_root = output / BASE_TOPIC
    pristine = root / ".pristine-topic"
    shutil.copytree(base_root, pristine)
    shutil.rmtree(base_root)
    docids: dict[str, str] = {}
    run_lines: list[str] = []
    provenance_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    depths: dict[str, dict[str, int]] = {}

    base_provenance = json.loads((output / "retrieval_provenance.jsonl").read_text(encoding="utf-8"))
    with zipfile.ZipFile(output / "retrieval_with_text.jsonl.zip") as archive:
        base_archive = json.loads(archive.read("retrieval_with_text.jsonl"))

    for index, spec in enumerate(specs):
        mapping = {
            BASE_TOPIC: spec.topic_id,
            BASE_NARRATIVE: spec.narrative,
            _sha256(BASE_NARRATIVE): _sha256(spec.narrative),
        }
        # Always clone the pristine base tree: cloning an already-rewritten topic would
        # leave the next topic's substitutions with nothing to match.
        topic_root = output / spec.topic_id
        shutil.copytree(pristine, topic_root)
        _rewrite_tree(topic_root, mapping)
        scoring_digest, canonical_digest = _reseal_topic(topic_root)

        docid = "doc-original"
        docids[spec.topic_id] = docid
        run_lines.append(f"{spec.topic_id} Q0 {docid} 1 1 {experiment_id}")
        row = rewrite(base_provenance, mapping)
        row["source_seals"]["scoring_manifest_sha256"] = scoring_digest
        row["source_seals"]["canonical_manifest_sha256"] = canonical_digest
        provenance_rows.append(row)
        archive_rows.append(rewrite(base_archive, mapping))
        depths[spec.topic_id] = {"official": 1, depth_kind: spec.candidate_documents}

    if shuffle_rows:
        # The declared order must survive inputs that arrive in a different order.
        provenance_rows.reverse()
        archive_rows.reverse()

    (output / "r_output_trec_rag_2026.tsv").write_bytes(
        "\n".join(run_lines).encode("utf-8") + b"\n"
    )
    (output / "retrieval_provenance.jsonl").write_bytes(
        b"".join(_json_bytes(row) for row in provenance_rows)
    )
    with zipfile.ZipFile(
        output / "retrieval_with_text.jsonl.zip", "w", compression=zipfile.ZIP_STORED
    ) as archive:
        archive.writestr(
            "retrieval_with_text.jsonl", b"".join(_json_bytes(row) for row in archive_rows)
        )

    manifest_path = output / "retrieval_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = experiment_id
    manifest["selected_topic_ids"] = [spec.topic_id for spec in specs]
    manifest["topic_depths"] = depths
    manifest["official_row_count"] = len(specs)
    for name in (
        "r_output_trec_rag_2026.tsv",
        "retrieval_provenance.jsonl",
        "retrieval_with_text.jsonl.zip",
    ):
        body = (output / name).read_bytes()
        manifest["artifacts"][name] = {"bytes": len(body), "sha256": sha256(body).hexdigest()}
    manifest_path.write_bytes(_json_bytes(manifest))

    rag_config, rag_output, run_id = _write_rag(root, output, specs, experiment_id)
    fixture = RunFixture(
        root=root,
        retrieval_config=retrieval_config,
        rag_config=rag_config,
        retrieval_output=output,
        rag_output=rag_output,
        work_dir=root / "work",
        specs=tuple(specs),
        experiment_id=experiment_id,
        run_id=run_id,
        docids=docids,
    )
    write_generation_identity(fixture)
    return fixture


def build_accepted_run(
    root: Path,
    specs: Sequence[TopicSpec],
    *,
    experiment_id: str = "debug-fixture",
) -> RunFixture:
    """Materialize a run plus an extracted accepted-submission metadata receipt."""
    fixture = build_run(root, specs, experiment_id=experiment_id)
    accepted_output = root / "accepted" / "rag_output_trec_rag_2026.jsonl"
    accepted_output.parent.mkdir(parents=True, exist_ok=True)
    accepted_output.write_bytes(fixture.rag_output.read_bytes())
    handoff = fixture.retrieval_output / "generation_handoff_manifest.json"
    handoff_payload = json.loads(handoff.read_text(encoding="utf-8"))
    metadata = root / "accepted-bundle-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": "trec-rag-2026-rag-submission-bundle-v1",
                "team_id": "fixture-team",
                "provider": "fixture-provider",
                "source": {
                    "generation_handoff_manifest_sha256": handoff_payload["manifest_sha256"],
                    "topic_count": len(specs),
                },
                "runs": [
                    {
                        "name": "accepted-fixture",
                        "run_id": fixture.run_id,
                        "run_desc": RUN_DESC,
                        "path": "accepted/rag_output_trec_rag_2026.jsonl",
                        "bytes": len(accepted_output.read_bytes()),
                        "line_count": len(specs),
                        "sha256": sha256(accepted_output.read_bytes()).hexdigest(),
                        "provider": "fixture-provider",
                        "models": ["fixture/model"],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return replace(
        fixture,
        rag_output=accepted_output,
        accepted_bundle_metadata=metadata,
        accepted_handoff=handoff,
    )


build_accepted_fixture = build_accepted_run


def _generation_topic(spec: TopicSpec, *, docid: str = "doc-original"):
    """Build a handoff topic whose evidence text is controlled by the spec."""
    from trec_rag.generation_handoff import EvidencePassage, EvidenceSourceSpan

    topic = _rag_generation_topic(spec.topic_id, spec.narrative, docid=docid)
    original = topic.evidence[0]
    passage = spec.evidence_text
    replaced = EvidencePassage(
        evidence_id=original.evidence_id,
        group_id=original.group_id,
        cluster_id=original.cluster_id,
        cluster_ordinal=original.cluster_ordinal,
        support_ordinal=original.support_ordinal,
        candidate_kind=original.candidate_kind,
        docid=original.docid,
        document_rank=original.document_rank,
        text=passage,
        document_sha256=original.document_sha256,
        source_span=EvidenceSourceSpan(
            start_char=0,
            end_char=len(passage),
            start_byte=0,
            end_byte=len(passage.encode("utf-8")),
        ),
    )
    return type(topic)(
        topic_id=topic.topic_id,
        narrative=topic.narrative,
        groups=topic.groups,
        evidence=(replaced,),
        claim_hints=topic.claim_hints,
        source_receipts=topic.source_receipts,
    )


def _write_rag(
    root: Path, output: Path, specs: Sequence[TopicSpec], experiment_id: str
) -> tuple[Path, Path, str]:
    _write_debug_handoff(output, tuple(_generation_topic(spec) for spec in specs))
    rag_experiment = f"rag-{experiment_id}"
    topic_list = ", ".join(spec.topic_id for spec in specs)
    rag_config = root / f"{rag_experiment}.yaml"
    rag_config.write_text(
        f"""schema_version: competition_rag_config_v2
experiment:
  id: {rag_experiment}
  output_dir: outputs/{rag_experiment}
  mode: create
  topic_ids: [{topic_list}]
submission:
  team_id: fixture-team
  run_desc: {RUN_DESC}
inputs:
  handoff_manifest: outputs/{experiment_id}/generation_handoff_manifest.json
generation:
  type: openrouter
  api_base: https://example.invalid/api/v1
  api_key_env: UNUSED_FIXTURE_KEY
  model: fixture/model
  reasoning_effort: medium
  temperature: null
  max_tokens: 1000
  timeout_seconds: 30
  transport_max_attempts: 1
  concurrency: 1
""",
        encoding="utf-8",
    )
    rag_output = root / "outputs" / rag_experiment / "rag_output_trec_rag_2026.jsonl"
    rag_output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(rag_output, specs, rag_experiment)
    return rag_config, rag_output, rag_experiment


def submission_records(
    specs: Sequence[TopicSpec], run_id: str, *, docid: str = "doc-original"
) -> list[dict[str, Any]]:
    return [
        {
            "metadata": {
                "team_id": "fixture-team",
                "narrative_id": spec.topic_id,
                "narrative": spec.narrative,
                "run_id": run_id,
                "run_desc": RUN_DESC,
            },
            "references": [docid],
            "answer": [
                {"text": text, "citations": list(citations)} for text, citations in spec.answers
            ],
        }
        for spec in specs
    ]


def write_submission(
    path: Path, specs: Sequence[TopicSpec], run_id: str, *, records: Sequence[dict] | None = None
) -> None:
    rows = list(records) if records is not None else submission_records(specs, run_id)
    path.write_bytes(
        b"".join(
            (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            for row in rows
        )
    )


def write_generation_identity(fixture: RunFixture) -> Path:
    """Write the generation identity the RAG runner would have produced."""
    from trec_rag.competition_rag import _generation_identity, load_rag_generation_config
    from trec_rag.generation_handoff import load_generation_handoff, select_generation_topics

    config = load_rag_generation_config(fixture.rag_config)
    handoff = load_generation_handoff(config.handoff_manifest_path)
    topics = select_generation_topics(handoff, config.topic_ids)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    path = config.work_dir / "generation_identity.json"
    path.write_text(json.dumps(_generation_identity(config, handoff, topics)), encoding="utf-8")
    return path
