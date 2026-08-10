"""Render the private friendly evaluation report from a validated bundle manifest.

The renderer reads one input — ``evaluation_manifest.json`` — and rebuilds the whole page
from a repository-owned template every time. Nothing is patched in place, so the same
manifest and the same code revision always produce byte-identical HTML.

Two layers keep private material out. First an allowlisted presentation model is built:
only narrative, subnarrative, response text, citation positions, support labels, safe
metric values, and summarized provenance survive into it. Then the rendered page is
scanned against generic secret patterns and a denylist derived from the actual private
inputs, and the page is discarded rather than written if anything forbidden appears.
"""

from __future__ import annotations

import html
import os
import re
import secrets
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trec_rag.offline_evaluation import (
    BUNDLE_SCHEMA_VERSION,
    SUPPORT_METRIC_PAIRS,
    EvaluationError,
)


REPORT_SCHEMA_VERSION = "trec_rag_friendly_evaluation_report_v1"
SUPPORT_LABELS = ("FS", "PS", "NS")
LABEL_NAMES = {"FS": "Full", "PS": "Partial", "NS": "None"}
LABEL_PHRASES = {"FS": "full support", "PS": "partial support", "NS": "no support"}

COUNT_DEFINITIONS = (
    ("submitted_documents", "submitted documents", "Documents retained in the organizer-facing retrieval run for this topic."),
    ("evidence_documents", "evidence documents", "Distinct documents in the authenticated selected-evidence handoff."),
    ("evidence_passages", "evidence passages", "Authenticated passages supplied to generation; one document can contribute several."),
    ("answer_references", "answer references", "Documents listed by the response; one reference can support several statements."),
    ("answer_objects", "answer objects", "Statements in the response."),
    ("citations", "citations", "Statement-citation pairs in the response."),
    ("judgments", "judgments", "Completed RAGDoll judgments, one per statement-citation pair."),
)

CANDIDATE_POOL_DEFINITIONS = {
    "natural_union": (
        "candidate documents (natural union)",
        "Deduplicated union across query lanes before final selection.",
    ),
    "candidate_pool": (
        "candidate documents (sealed pool)",
        "Documents in the sealed candidate pool before final selection.",
    ),
}

COUNT_KEYS = ("candidate_documents", *(key for key, _label, _definition in COUNT_DEFINITIONS))

_PRIVATE_PATTERNS = (
    (
        r"(?<![0-9a-fA-F])(?=[0-9a-fA-F]{12,}(?![0-9a-fA-F]))"
        r"(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{12,}(?![0-9a-fA-F])",
        "a long hexadecimal token",
    ),
    (r"\bsk-[A-Za-z0-9]{8,}", "a credential"),
    (r"OPENROUTER|PYSERINI_API_TOKEN|INDEX_URL|API_KEY", "a credential name"),
    (r"\bdeepagent[-_/]", "DeepAgent material"),
    (r"\braw_output\b|\bfinish_reason\b|\bprompt_tokens\b", "a provider event field"),
    (r"(?<![\w.])/(?:tmp|home|Users|var/folders)/", "a private filesystem path"),
)


class ReportPrivacyError(EvaluationError):
    """Raised when the assembled page would publish forbidden material."""


# ---------------------------------------------------------------------------
# Presentation model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Presentation:
    """The allowlisted view model. Nothing else may reach the template."""

    schema_version: str
    topic_ids: tuple[str, ...]
    topics: tuple[Mapping[str, Any], ...]
    totals: Mapping[str, int]
    label_counts: Mapping[str, int]
    citation_support: Mapping[str, Any]
    retrieval: Mapping[str, Any]
    nuggets: Mapping[str, Any]
    metric_definitions: Mapping[str, Any]
    provenance: Mapping[str, Any]
    validation: Mapping[str, Any]
    commands: tuple[str, ...]
    answer_word_limit: int


def build_presentation(manifest: Mapping[str, Any], *, commands: Sequence[str] = ()) -> Presentation:
    """Validate the manifest and project only publishable values out of it."""
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise EvaluationError(f"manifest schema is not {BUNDLE_SCHEMA_VERSION}")
    ordered = tuple(manifest["scope"]["topic_ids"])
    if not ordered or len(set(ordered)) != len(ordered):
        raise EvaluationError("manifest scope has no topics or repeats one")

    raw_topics = manifest["topics"]
    if [str(topic["topic_id"]) for topic in raw_topics] != list(ordered):
        raise EvaluationError("manifest topic order does not match the declared scope")

    support = manifest["metrics"]["citation_support"]
    label_counts_per_topic = manifest["judgments"]["label_counts_per_topic"]

    topics: list[Mapping[str, Any]] = []
    totals = {key: 0 for key in COUNT_KEYS}
    for raw in raw_topics:
        topic_id = str(raw["topic_id"])
        candidate_pool_kind = str(raw.get("candidate_pool_kind", ""))
        if candidate_pool_kind not in CANDIDATE_POOL_DEFINITIONS:
            raise EvaluationError(
                f"{topic_id}: invalid candidate_pool_kind {candidate_pool_kind!r}"
            )
        counts = dict(raw["counts"])
        labels = dict(label_counts_per_topic.get(topic_id, {}))
        counts["judgments"] = sum(labels.values())
        answer = []
        for index, item in enumerate(raw["answer"], start=1):
            citations = []
            for citation in item["citations"]:
                label = citation.get("support_label")
                if label is not None and label not in SUPPORT_LABELS:
                    raise EvaluationError(f"{topic_id}: invalid support label {label!r}")
                position = citation.get("position")
                if not isinstance(position, int) or position < 1:
                    raise EvaluationError(f"{topic_id}: citation has no reference position")
                citations.append({"position": position, "support_label": label})
            answer.append({"index": index, "text": str(item["text"]), "citations": citations})
        if counts["citations"] != sum(len(item["citations"]) for item in answer):
            raise EvaluationError(f"{topic_id}: citation count disagrees with the response")
        for key in totals:
            totals[key] += int(counts.get(key, 0))
        topics.append(
            {
                "topic_id": topic_id,
                "candidate_pool_kind": candidate_pool_kind,
                "narrative": str(raw["narrative"]),
                "subnarratives": [str(text) for text in raw["subnarratives"]],
                "answer": answer,
                "counts": counts,
                "label_counts": labels,
                "support": {
                    "metrics": support["per_topic"].get(topic_id, {}),
                    "availability": support["per_topic_availability"].get(topic_id, {}),
                },
                "retrieval": {
                    "metrics": manifest["metrics"]["retrieval"]["per_topic"].get(topic_id, {}),
                    "availability": manifest["metrics"]["retrieval"]["per_topic_availability"].get(
                        topic_id, {}
                    ),
                },
                "nuggets": manifest["metrics"]["nugget_coverage"]["per_topic_availability"].get(
                    topic_id, {}
                ),
            }
        )

    label_counts = dict(manifest["judgments"]["label_counts"])
    if sum(label_counts.values()) != totals["judgments"]:
        raise EvaluationError("aggregate label counts disagree with the per-topic counts")

    judge = manifest["judge"]
    ragdoll = manifest["ragdoll"]
    identities = manifest["identities"]
    source_identity_available = identities.get("source_identity_available", True)
    if type(source_identity_available) is not bool:
        raise EvaluationError("identities.source_identity_available must be a boolean")
    source_identity_reason = identities.get("source_identity_reason")
    if source_identity_reason is not None and not isinstance(source_identity_reason, str):
        raise EvaluationError("identities.source_identity_reason must be text or null")
    provenance = {
        "ragdoll_version": str(ragdoll["project_version"]),
        "ragdoll_commit": str(ragdoll["commit"])[:7],
        "judge_provider": str(judge["provider"]),
        "judge_model": str(judge["model"]),
        "judge_thinking": str(judge["thinking"]),
        "judge_tasks": int(judge["tasks"]),
        "hosted_calls": int(judge["hosted_calls"]),
        "reused_from_cache": int(judge["reused_from_cache"]),
        "cache": {key: int(value) for key, value in manifest["cache"].items()},
        "run_identity_sha256": str(manifest["identities"]["generation_identity_sha256"])[:7],
        "source_identity_available": source_identity_available,
        "source_identity_reason": source_identity_reason,
    }
    return Presentation(
        schema_version=REPORT_SCHEMA_VERSION,
        topic_ids=ordered,
        topics=tuple(topics),
        totals=totals,
        label_counts=label_counts,
        citation_support={
            "macro": support["macro"],
            "macro_availability": support["macro_availability"],
        },
        retrieval={
            "macro": manifest["metrics"]["retrieval"]["macro"],
            "macro_availability": manifest["metrics"]["retrieval"]["macro_availability"],
        },
        nuggets=manifest["metrics"]["nugget_coverage"]["macro_availability"],
        metric_definitions=manifest["metric_definitions"],
        provenance=provenance,
        validation=dict(manifest["validation"]),
        commands=tuple(str(command) for command in commands),
        answer_word_limit=int(manifest["contract"]["answer_word_limit"]),
    )


# ---------------------------------------------------------------------------
# Privacy denylist
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrivacyDenylist(Sequence[str]):
    """Separate always-forbidden identifiers from collision-safe passage text."""

    identifiers: tuple[str, ...]
    passage_texts: tuple[str, ...]

    def _values(self) -> tuple[str, ...]:
        return self.identifiers + self.passage_texts

    def __len__(self) -> int:
        return len(self.identifiers) + len(self.passage_texts)

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        return self._values()[index]


def denylist_from_bundle(work_dir: Path, extra: Iterable[str] = ()) -> PrivacyDenylist:
    """Collect concrete private values, separating identifiers from passages.

    The values come from the bundle's own derived artifacts — document identifiers,
    evidence passages, prompts, and full digests — so the scan catches a leak of
    this run's real data rather than only matching generic regexes. Identifiers
    are always scanned against the unredacted page; passage text may recur in the
    explicitly allowlisted response/narrative fields.
    """
    identifiers: set[str] = {
        value for value in extra if isinstance(value, str) and value.strip()
    }
    passage_texts: set[str] = set()
    support_input = Path(work_dir) / "support_input.jsonl"
    if support_input.is_file():
        import json

        for line in support_input.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for docid in row.get("references", []):
                identifiers.add(str(docid))
            for docid, text in (row.get("segments") or {}).items():
                identifiers.add(str(docid))
                passage_texts.add(str(text))
    return PrivacyDenylist(
        identifiers=tuple(sorted(value for value in identifiers if len(value) >= 6)),
        passage_texts=tuple(sorted(value for value in passage_texts if len(value) >= 6)),
    )


_BLOCK_HTML_TAGS = re.compile(
    r"</?(?:address|article|aside|blockquote|br|dd|details|div|dl|dt|footer|form|h[1-6]|header|hr|li|main|nav|ol|p|pre|section|summary|table|tr|ul)\b[^>]*>",
    flags=re.IGNORECASE,
)


def _privacy_projections(page: str) -> tuple[str, str, str]:
    decoded = html.unescape(page)
    plain = _BLOCK_HTML_TAGS.sub("\n", decoded)
    plain = re.sub(r"<[^>]+>", "", plain)
    return page, decoded, plain


def _denylist_values(denylist: Sequence[str] | PrivacyDenylist) -> tuple[str, ...]:
    if isinstance(denylist, PrivacyDenylist):
        return denylist._values()
    return tuple(denylist)


def assert_publishable(
    page: str, denylist: Sequence[str] | PrivacyDenylist = ()
) -> None:
    projections = _privacy_projections(page)
    for pattern, description in _PRIVATE_PATTERNS:
        for projection in projections:
            match = re.search(pattern, projection, flags=re.IGNORECASE)
            if match is not None:
                raise ReportPrivacyError(
                    f"refusing to write the report: it contains {description} ({match.group(0)[:24]!r})"
                )
    normalized_projections = tuple(" ".join(projection.split()) for projection in projections)
    for value in _denylist_values(denylist):
        normalized_value = " ".join(value.split()) if value else ""
        if value and (
            any(value in projection for projection in projections)
            or any(normalized_value in projection for projection in normalized_projections)
        ):
            raise ReportPrivacyError(
                f"refusing to write the report: it contains a private input value "
                f"({value[:24]!r})"
            )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _inline(value: str) -> str:
    rendered = html.escape(value.strip(), quote=True)
    rendered = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", rendered)
    return rendered.replace("\n", "<br>")


JUDGMENT_EXCERPT_CHARACTERS = 160


def _excerpt(text: str, limit: int = JUDGMENT_EXCERPT_CHARACTERS) -> str:
    """Quote the judged response statement so a label identifies its own claim.

    Response text is already published in full above, so this adds no new disclosure. It
    is deterministically truncated on a word boundary to keep the list scannable.
    """
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return _escape(collapsed)
    head = collapsed[:limit].rsplit(" ", 1)[0].rstrip(",;:.")
    return _escape(f"{head}…")


def _paired(metrics: Mapping[str, Any], precision_key: str, recall_key: str) -> tuple[str, str]:
    """Collapse precision and recall into one number only after proving they match."""
    precision = metrics.get(precision_key)
    recall = metrics.get(recall_key)
    if precision is None or recall is None:
        return "—", "unavailable"
    if abs(float(precision) - float(recall)) <= 1e-9:
        return f"{float(precision):.6f}", "precision / recall"
    return f"P {float(precision):.6f} · R {float(recall):.6f}", "precision and recall differ"


def _metric_title(precision_key: str) -> str:
    return {
        "weighted_precision_first_citation": "Weighted first-citation",
        "weighted_precision_all_judged_citations": "Weighted all-citation",
        "hard_precision": "Hard first-citation",
    }[precision_key]


def _availability_row(title: str, availability: Mapping[str, Any]) -> str:
    available = bool(availability.get("available"))
    status = "Available" if available else "Not available"
    css = "status-available" if available else "status-na"
    reason = availability.get("reason") or "Required labels are complete."
    return (
        f'        <div class="availability-row"><h3>{_escape(title)}</h3>'
        f'<span class="{css}">{status}</span><p>{_escape(reason)}</p></div>'
    )


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


def render_report(view: Presentation) -> str:
    """Build the whole standalone page deterministically from the view model."""
    body = "\n".join(
        [
            _render_hero(view),
            _render_glance(view),
            _render_retrieval(view),
            _render_answers(view),
            _render_topics(view),
            _render_provenance(view),
        ]
    )
    return _PAGE.format(style=_STYLE, body=body)


def _privacy_scan_view(view: Presentation) -> Presentation:
    """Clone the model and redact only text fields explicitly published by the template."""
    scan_view = deepcopy(view)
    sentinel = f"friendly-report-allowlisted-{secrets.token_urlsafe(24)}"
    for topic in scan_view.topics:
        topic["narrative"] = sentinel
        topic["subnarratives"] = [sentinel for _ in topic["subnarratives"]]
        for answer in topic["answer"]:
            answer["text"] = sentinel
    return scan_view


def _render_hero(view: Presentation) -> str:
    labels = view.label_counts
    judged = view.totals["judgments"]
    return "\n".join(
        [
            '    <header class="hero">',
            '      <div class="hero-inner">',
            '        <p class="eyebrow">Private post-run evaluation</p>',
            "        <h1>Retrieval and answer evaluation</h1>",
            f'        <p class="lede">{len(view.topic_ids)} topic(s) in declared order. '
            f"RAGDoll judged {judged} statement-citation pair(s): "
            f'{labels.get("FS", 0)} full support, {labels.get("PS", 0)} partial support, and '
            f'{labels.get("NS", 0)} no support.</p>',
            '        <div class="scope" aria-label="Evaluation scope">',
            f'          <span class="tag">{len(view.topic_ids)} topics</span>',
            f'          <span class="tag">{view.provenance["judge_tasks"]} judge tasks</span>',
            f'          <span class="tag">{view.provenance["hosted_calls"]} hosted calls</span>',
            f'          <span class="tag">RAGDoll {_escape(view.provenance["ragdoll_version"])}</span>',
            "        </div>",
            "      </div>",
            '      <div class="hero-note"><strong>Important:</strong> “Not available” is not a zero. '
            "A metric whose required labels are incomplete is reported as unavailable with its exact "
            "limitation, never as a score.</div>",
            "    </header>",
        ]
    )


def _render_glance(view: Presentation) -> str:
    cards = [
        (f'{view.totals["submitted_documents"]:,}', "submitted retrieval documents"),
        (f'{view.totals["answer_objects"]:,}', "answer objects"),
        (f'{view.totals["citations"]:,}', "statement-citation pairs"),
        (f'{view.totals["judgments"]:,}', "completed judgments"),
    ]
    candidate_kinds = {str(topic["candidate_pool_kind"]) for topic in view.topics}
    if len(candidate_kinds) == 1:
        candidate_label, candidate_definition = CANDIDATE_POOL_DEFINITIONS[
            next(iter(candidate_kinds))
        ]
    else:
        candidate_label = "candidate documents (mixed pool kinds)"
        candidate_definition = (
            "Sum of differently defined per-topic candidate counts; open each topic for its kind."
        )
    definitions = "\n".join(
        f'          <article><strong>{view.totals[key]:,} {_escape(label)}</strong>'
        f"<p>{_escape(definition)}</p></article>"
        for key, label, definition in (
            ("candidate_documents", candidate_label, candidate_definition),
            *COUNT_DEFINITIONS,
        )
    )
    return "\n".join(
        [
            '    <section class="section" aria-labelledby="glance-heading">',
            '      <div class="section-head"><div><h2 id="glance-heading">At a glance</h2>',
            "        <p>Pipeline volumes across the selected scope. These are diagnostics, not "
            "relevance or correctness judgments.</p></div></div>",
            '      <div class="stats">',
            *[
                f'        <div class="stat"><span class="stat-value">{value}</span>'
                f'<span class="stat-label">{_escape(label)}</span></div>'
                for value, label in cards
            ],
            "      </div>",
            '      <details class="count-guide"><summary>How to read these counts</summary>',
            '        <div class="count-guide-grid">',
            definitions,
            "        </div>",
            "      </details>",
            "    </section>",
        ]
    )


def _render_retrieval(view: Presentation) -> str:
    rows = [_availability_row("Retrieval relevance (qrels)", view.retrieval["macro_availability"])]
    cards: list[str] = []
    for key, value in sorted(view.retrieval["macro"].items()):
        definition = view.metric_definitions["retrieval"].get(key, "")
        cards.append(
            f'        <article class="metric-card"><strong>{float(value):.6f}</strong>'
            f"<span>{_escape(key)}</span><p>{_escape(definition)}</p></article>"
        )
    per_topic = "\n".join(
        f'        <div class="availability-row"><h3>{_escape(topic["topic_id"])}</h3>'
        f'<span class="{"status-available" if topic["retrieval"]["availability"].get("available") else "status-na"}">'
        f'{"Available" if topic["retrieval"]["availability"].get("available") else "Not available"}</span>'
        f'<p>{_escape(topic["retrieval"]["availability"].get("reason") or "Matching qrels are complete.")}</p></div>'
        for topic in view.topics
    )
    return "\n".join(
        [
            '    <section class="section" aria-labelledby="retrieval-heading">',
            '      <div class="section-head"><div><h2 id="retrieval-heading">Retrieval scores</h2>',
            "        <p>Ranked-retrieval metrics require qrels that match the selected topics. "
            "They are kept separate from the answer and citation metrics below.</p>"
            "</div></div>",
            '      <div class="availability">',
            *rows,
            per_topic,
            "      </div>",
            *(
                ['      <div class="metric-grid" aria-label="Macro retrieval metrics">', *cards, "      </div>"]
                if cards
                else []
            ),
            *_render_per_topic_metric_table(
                view,
                key="retrieval",
                caption="Per-topic retrieval metrics",
                identifier="retrieval-per-topic",
            ),
            "    </section>",
        ]
    )


def _render_per_topic_metric_table(
    view: Presentation, *, key: str, caption: str, identifier: str
) -> list[str]:
    """Render one row per topic and one column per metric the manifest actually carries.

    Columns are the union of the metric names present across available topics, so nothing
    is hardcoded. A topic without values keeps an explicit unavailable cell rather than a
    blank that could be mistaken for a zero.
    """
    available = [topic for topic in view.topics if topic[key]["metrics"]]
    if not available:
        return []
    columns = sorted({name for topic in available for name in topic[key]["metrics"]})
    header = "".join(f"<th scope=\"col\">{_escape(name)}</th>" for name in columns)
    body: list[str] = []
    for topic in view.topics:
        metrics = topic[key]["metrics"]
        cells = []
        for name in columns:
            if name in metrics:
                cells.append(f"<td>{float(metrics[name]):.6f}</td>")
            else:
                reason = topic[key]["availability"].get("reason") or "not available"
                cells.append(
                    f'<td class="cell-na"><span class="visually-hidden">{_escape(reason)}</span>'
                    "<span aria-hidden=\"true\">Not available</span></td>"
                )
        body.append(
            f'          <tr><th scope="row">{_escape(topic["topic_id"])}</th>{"".join(cells)}</tr>'
        )
    return [
        f'      <div class="table-scroll">',
        f'        <table class="metric-table" aria-labelledby="{identifier}-caption">',
        f'          <caption id="{identifier}-caption">{_escape(caption)}</caption>',
        f'          <thead><tr><th scope="col">Topic</th>{header}</tr></thead>',
        "          <tbody>",
        *body,
        "          </tbody>",
        "        </table>",
        "      </div>",
    ]


def _render_answers(view: Presentation) -> str:
    support = view.citation_support
    cards: list[str] = []
    for precision_key, recall_key in SUPPORT_METRIC_PAIRS:
        value, caption = _paired(support["macro"], precision_key, recall_key)
        definitions = view.metric_definitions["citation_support"]
        # Precision and recall share a numerator but not a denominator, so publish both
        # definitions even when the two values happen to be equal.
        detail = " ".join(
            text
            for text in (definitions.get(precision_key, ""), definitions.get(recall_key, ""))
            if text
        )
        cards.append(
            f'        <article class="metric-card"><strong>{value}</strong>'
            f"<span>{_escape(_metric_title(precision_key))} {_escape(caption)}</span>"
            f"<p>{_escape(detail)}</p></article>"
        )
    labels = view.label_counts
    cards.append(
        f'        <article class="metric-card"><strong>{labels.get("FS", 0)} · '
        f'{labels.get("PS", 0)} · {labels.get("NS", 0)}</strong>'
        "<span>FS · PS · NS labels across the selected scope</span>"
        "<p>Every statement-citation pair is judged independently.</p></article>"
    )
    return "\n".join(
        [
            '    <section class="section" aria-labelledby="answers-heading">',
            '      <div class="section-head"><div><h2 id="answers-heading">Answer and citation scores</h2>',
            "        <p>Citation support is computed by the pinned RAGDoll implementation, never "
            "reimplemented here.</p></div></div>",
            '      <div class="availability">',
            _availability_row("Citation support", support["macro_availability"]),
            _availability_row("Nugget coverage", view.nuggets),
            "      </div>",
            '      <div class="metric-grid" aria-label="Macro citation-support metrics">',
            *cards,
            "      </div>",
            f'      <p class="caption">{_escape(view.metric_definitions["macro"])}</p>',
            "    </section>",
        ]
    )


def _render_topics(view: Presentation) -> str:
    return "\n".join(
        [
            '    <section class="section" aria-labelledby="topics-heading">',
            '      <div class="section-head"><div><h2 id="topics-heading">Topic explorer</h2>',
            "        <p>Open a topic for its narrative, subnarratives, response, and exact judgment "
            "status.</p></div></div>",
            '      <nav class="detail-intro" aria-label="Jump to topic details">',
            *[
                f'        <a class="detail-jump" href="#details-{_escape(topic["topic_id"])}">'
                f'{_escape(topic["topic_id"])}</a>'
                for topic in view.topics
            ],
            "      </nav>",
            *[
                _render_topic(view, topic, open_topic=index == 0)
                for index, topic in enumerate(view.topics)
            ],
            "    </section>",
        ]
    )


def _render_topic(view: Presentation, topic: Mapping[str, Any], *, open_topic: bool) -> str:
    topic_id = _escape(topic["topic_id"])
    counts = topic["counts"]
    responses: list[str] = []
    judgments: list[str] = []
    for item in topic["answer"]:
        parts: list[str] = []
        seen: list[str] = []
        for citation in item["citations"]:
            position = citation["position"]
            label = citation["support_label"]
            if label is None:
                parts.append(
                    f'<span class="citation citation-unjudged">[{position}] '
                    '<span class="citation-support">—</span>'
                    '<span class="visually-hidden"> not judged</span></span>'
                )
                judgments.append(
                    '              <li class="judgment-task">'
                    f'<span class="judgment-claim">Answer {item["index"]} · reference '
                    f'[{position}]<q class="judgment-statement">{_excerpt(item["text"])}</q></span>'
                    '<span class="judgment-badge label-none">Not judged</span></li>'
                )
                continue
            phrase = LABEL_PHRASES[label]
            seen.append(label.lower())
            parts.append(
                f'<span class="citation citation-judged label-{label.lower()}" '
                f'title="Reference {position}: {phrase[:1].upper()}{phrase[1:]}">'
                f'[{position}] <span class="citation-support">{label}</span>'
                f'<span class="visually-hidden"> — {phrase}</span></span>'
            )
            judgments.append(
                '              <li class="judgment-task">'
                f'<span class="judgment-claim">Answer {item["index"]} · reference '
                f'[{position}]<q class="judgment-statement">{_excerpt(item["text"])}</q></span>'
                f'<span class="judgment-badge label-{label.lower()}">'
                f"{label} · {LABEL_NAMES[label]}</span></li>"
            )
        classes = " ".join(f"response-has-{name}" for name in sorted(set(seen)))
        responses.append(
            f'          <div class="response-item {classes}">'
            f'<span class="response-number">{item["index"]}</span>'
            f'<p class="response-text">{_inline(item["text"])}'
            f'<span class="citations">{"".join(parts)}</span></p></div>'
        )

    metric_cards: list[str] = []
    for precision_key, recall_key in SUPPORT_METRIC_PAIRS:
        value, caption = _paired(topic["support"]["metrics"], precision_key, recall_key)
        metric_cards.append(
            f"              <article><span>{_escape(_metric_title(precision_key))} "
            f"{_escape(caption)}</span><strong>{value}</strong></article>"
        )

    labels = topic["label_counts"]
    candidate_kind = str(topic["candidate_pool_kind"])
    candidate_stage = (
        f'{counts.get("candidate_documents", 0):,} union candidates'
        if candidate_kind == "natural_union"
        else f'{counts.get("candidate_documents", 0):,} pool candidates'
    )
    stages = [
        ("Plan", f'{len(topic["subnarratives"])} subnarratives'),
        ("Search", candidate_stage),
        ("Submit", f'{counts.get("submitted_documents", 0):,} documents'),
        (
            "Evidence",
            f'{counts.get("evidence_documents", 0):,} docs · {counts.get("evidence_passages", 0):,} passages',
        ),
        (
            "Response",
            f'{counts.get("answer_objects", 0)} objects · {counts.get("answer_references", 0)} references · '
            f'{counts.get("answer_words", 0):,} words',
        ),
        ("Judge", f'{counts.get("judgments", 0)} completed'),
    ]
    return "\n".join(
        [
            f'      <details class="topic-disclosure" id="details-{topic_id}"'
            f'{" open" if open_topic else ""}>',
            "        <summary>",
            f'          <h3>{topic_id} <span class="summary-meta">· {len(topic["subnarratives"])} '
            f'subnarratives · {counts.get("answer_objects", 0)} answer objects</span></h3>',
            "        </summary>",
            '        <div class="topic-detail-body">',
            '          <ol class="stage-strip" aria-label="Pipeline stages">',
            *[
                f'            <li><span class="stage-name">{_escape(name)}</span>'
                f"<strong>{_escape(value)}</strong></li>"
                for name, value in stages
            ],
            "          </ol>",
            '          <div class="source-block"><h4>Narrative</h4>',
            f'            <blockquote class="narrative">{_inline(topic["narrative"])}</blockquote>',
            "          </div>",
            '          <div class="source-block"><h4>Subnarratives</h4>',
            '            <details class="subnarrative-disclosure">',
            f'              <summary>{len(topic["subnarratives"])} generated subnarratives</summary>',
            '              <ol class="subnarratives">',
            *[
                f'                <li class="subnarrative"><div class="subnarrative-head">'
                f'<p class="subnarrative-text">{_inline(text)}</p></div></li>'
                for text in topic["subnarratives"]
            ],
            "              </ol>",
            "            </details>",
            "          </div>",
            '          <div class="source-block"><h4>Our response</h4>',
            '            <p class="inline-support-note"><strong>Inline support:</strong> each reference '
            'is judged independently — <span class="mini-label label-fs">FS</span> full support, '
            '<span class="mini-label label-ps">PS</span> partial support, '
            '<span class="mini-label label-ns">NS</span> no support.</p>',
            '            <div class="response">',
            *responses,
            "            </div>",
            '            <p class="privacy-note">Citation numbers preserve the response’s reference '
            "positions. Document identifiers and retrieved passages are omitted from this view.</p>",
            "          </div>",
            f'          <div class="source-block judgment-block" id="judgments-{topic_id}">',
            "            <h4>Evaluation and judgments</h4>",
            '            <div class="judgment-summary">',
            '              <article class="judgment-verified"><span class="judgment-label">Citation '
            f'support</span><strong>{counts.get("judgments", 0)} judged</strong>'
            f'<p>{labels.get("FS", 0)} full · {labels.get("PS", 0)} partial · '
            f'{labels.get("NS", 0)} none.</p></article>',
            '              <article><span class="judgment-label">Nugget coverage</span>'
            "<strong>Not judged</strong>"
            f'<p>{_escape(topic["nuggets"].get("reason") or "No completed nugget assignments.")}</p>'
            "</article>",
            "            </div>",
            '            <div class="support-metrics" aria-label="Citation-support metrics">',
            *metric_cards,
            "            </div>",
            '            <details class="judgment-tasks">',
            f'              <summary>Citation-support judgments ({len(judgments)})</summary>',
            '              <ul class="judgment-task-list">',
            *judgments,
            "              </ul>",
            "            </details>",
            "          </div>",
            "        </div>",
            "      </details>",
        ]
    )


def _render_provenance(view: Presentation) -> str:
    provenance = view.provenance
    checks = "\n".join(
        f'        <li>{_escape(name.replace("_", " ").capitalize())}: '
        f'{"yes" if value else "no"}</li>'
        for name, value in sorted(view.validation.items())
    )
    commands = "\n".join(
        f"        <li><code>{_escape(command)}</code></li>" for command in view.commands
    )
    return "\n".join(
        [
            '    <section class="section" aria-labelledby="provenance-heading">',
            '      <div class="section-head"><div><h2 id="provenance-heading">Provenance and '
            "reproducibility</h2>",
            "        <p>Pinned versions, validation state, and cache behaviour.</p></div></div>",
            '      <ul class="checklist">',
            checks,
            "      </ul>",
            *(
                [
                    "      <h3>Command shape</h3>",
                    '      <p class="caption">Repo-relative and portable, not the exact '
                    "invocation: private work, cache, qrels, and output paths and the judge "
                    "options are deliberately omitted here and kept in the private run "
                    "receipt.</p>",
                    '      <ul class="commands">',
                    commands,
                    "      </ul>",
                ]
                if view.commands
                else []
            ),
            '      <div class="provenance">',
            f'        <span>RAGDoll <strong>{_escape(provenance["ragdoll_version"])}</strong> · '
            f'<code>{_escape(provenance["ragdoll_commit"])}</code></span>',
            f'        <span>Judge · <strong>{_escape(provenance["judge_provider"])} '
            f'{_escape(provenance["judge_model"])}</strong></span>',
            f'        <span>Judge tasks · <strong>{provenance["judge_tasks"]}</strong></span>',
            f'        <span>Hosted calls · <strong>{provenance["hosted_calls"]}</strong></span>',
            f'        <span>Reused from cache · <strong>{provenance["reused_from_cache"]}</strong></span>',
            *(
                [
                    "        <p class=\"caption\">Original generation identity unavailable — "
                    f'{_escape(provenance["source_identity_reason"] or "reason not recorded")}.</p>',
                ]
                if not provenance["source_identity_available"]
                else []
            ),
            "      </div>",
            "    </section>",
        ]
    )


def write_report(
    manifest: Mapping[str, Any],
    output_path: Path,
    *,
    denylist: Sequence[str] | PrivacyDenylist = (),
    commands: Sequence[str] = (),
) -> Path:
    """Render, scan, and only then write — atomically, so a failure never corrupts a
    previously valid report."""
    view = build_presentation(manifest, commands=commands)
    page = render_report(view)
    if isinstance(denylist, PrivacyDenylist):
        identifiers = denylist.identifiers
        passage_texts = denylist.passage_texts
    else:
        # Keep legacy callers fail-closed while the bundle API uses the split type.
        identifiers = tuple(denylist)
        passage_texts = identifiers
    # Generic patterns must inspect the actual page: allowlisted response text is still
    # forbidden when it contains credentials, paths, hashes, or provider/runtime fields.
    # Identifiers are always forbidden and therefore scan every projection of the real page.
    assert_publishable(page, denylist=identifiers)
    # Passage values are checked against a second render from a model-level projection.
    # The same renderer preserves every tag, attribute, and template/chrome field while
    # replacing only the exact allowlisted presentation-model text fields.
    scan_page = render_report(_privacy_scan_view(view))
    assert_publishable(scan_page, denylist=passage_texts)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = page.encode("utf-8")
    temporary = output_path.with_name(f"{output_path.name}.tmp-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    output_path.chmod(0o600)
    return output_path


_STYLE = """
    :root {
      --bg: #eef3f8; --surface: #ffffff; --surface-2: #f7f9fc; --ink: #172033;
      --muted: #4a5870; --line: #d8e1ec; --blue: #2563eb; --blue-soft: #dbeafe;
      --green: #15803d; --green-soft: #dcfce7; --amber: #9a5b00; --amber-soft: #fff4d6;
      --red: #b42318; --red-soft: #fee4e2; --gray-soft: #edf1f6;
      --badge-blue: #1d4ed8; --badge-green: #166534; --badge-amber: #8a5100; --badge-red: #991b1b;
      --shadow: 0 18px 48px rgb(32 48 75 / 9%);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      color: var(--ink); background: var(--bg);
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: var(--bg); }
    main { width: min(100% - 2rem, 70rem); margin: 0 auto; padding: 2rem 0 4rem; }
    a { color: var(--blue); }
    .hero, .section { background: var(--surface); border: 1px solid var(--line);
      border-radius: 1.25rem; box-shadow: var(--shadow); margin-top: 1rem; }
    .hero { overflow: hidden; margin-top: 0; }
    .hero-inner { padding: clamp(1.5rem, 5vw, 3.5rem); }
    .section { padding: clamp(1.3rem, 4vw, 2.2rem); }
    .eyebrow { display: flex; align-items: center; gap: .55rem; margin: 0 0 .85rem;
      color: var(--blue); font-size: .78rem; font-weight: 800; letter-spacing: .09em;
      text-transform: uppercase; }
    .eyebrow::before { content: ""; width: .7rem; height: .7rem; border-radius: 50%;
      background: var(--green); box-shadow: 0 0 0 .25rem var(--green-soft); }
    h1 { margin: 0; max-width: 20ch; font-size: clamp(2.1rem, 7vw, 4rem); line-height: 1;
      letter-spacing: -.045em; }
    h2 { margin: 0; font-size: clamp(1.45rem, 4vw, 2rem); letter-spacing: -.025em; }
    h3 { margin: 0; font-size: 1rem; }
    .lede { max-width: 46rem; margin: 1.25rem 0 0; color: var(--muted);
      font-size: clamp(1rem, 2vw, 1.2rem); line-height: 1.65; }
    .scope { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.35rem; }
    .tag { padding: .45rem .7rem; border: 1px solid var(--line); border-radius: 999px;
      background: var(--surface-2); color: var(--muted); font-size: .85rem; font-weight: 700; }
    .hero-note { padding: 1.1rem clamp(1.5rem, 5vw, 3.5rem); border-top: 1px solid #f0cf8b;
      background: var(--amber-soft); color: var(--amber); line-height: 1.55; }
    .section-head { margin-bottom: 1.4rem; }
    .section-head p, .caption { margin: .35rem 0 0; color: var(--muted); line-height: 1.55; }
    .stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; }
    .stat { padding: 1rem; background: var(--surface-2); border: 1px solid var(--line);
      border-radius: .9rem; }
    .stat-value { display: block; font-size: clamp(1.6rem, 4vw, 2.3rem); font-weight: 800;
      letter-spacing: -.04em; }
    .stat-label { display: block; margin-top: .22rem; color: var(--muted); font-size: .82rem;
      line-height: 1.35; }
    .availability { display: grid; gap: .7rem; margin-bottom: 1rem; }
    .availability-row { display: grid; grid-template-columns: 14rem 7rem 1fr; gap: 1rem;
      align-items: start; padding: 1rem 0; border-top: 1px solid var(--line); }
    .availability-row:first-child { border-top: 0; padding-top: 0; }
    .availability-row p { margin: 0; color: var(--muted); line-height: 1.5; }
    .status-na { justify-self: start; padding: .25rem .55rem; border-radius: .5rem;
      background: var(--gray-soft); color: var(--muted); font-size: .78rem; font-weight: 800; }
    .status-available { justify-self: start; padding: .25rem .55rem; border-radius: .5rem;
      background: var(--green-soft); color: var(--badge-green); font-size: .78rem; font-weight: 800; }
    .metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .7rem;
      margin-top: 1rem; }
    .metric-card { padding: 1rem; border: 1px solid var(--line); border-radius: .85rem;
      background: var(--surface-2); }
    .metric-card strong { display: block; font-size: 1.35rem; font-variant-numeric: tabular-nums;
      letter-spacing: -.025em; }
    .metric-card span { display: block; margin-top: .25rem; color: var(--muted); font-size: .78rem;
      line-height: 1.4; }
    .metric-card p { margin: .3rem 0 0; color: var(--muted); font-size: .74rem; line-height: 1.45; }
    .count-guide { border: 1px solid var(--line); border-radius: .8rem; background: var(--surface-2);
      margin-top: 1rem; }
    .count-guide > summary { padding: .82rem 1rem; cursor: pointer; color: var(--blue);
      font-size: .84rem; font-weight: 850; }
    .count-guide-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .55rem;
      padding: .85rem; }
    .count-guide-grid article { padding: .68rem .72rem; border: 1px solid var(--line);
      border-radius: .65rem; background: var(--surface); }
    .count-guide-grid strong { display: block; font-size: .8rem; line-height: 1.35; }
    .count-guide-grid p { margin: .28rem 0 0; color: var(--muted); font-size: .74rem; line-height: 1.45; }
    .detail-intro { display: flex; flex-wrap: wrap; gap: .55rem; margin-bottom: 1rem; }
    .detail-jump { display: inline-flex; padding: .48rem .72rem; border: 1px solid var(--line);
      border-radius: .6rem; background: var(--surface-2); color: var(--blue); font-size: .85rem;
      font-weight: 750; text-decoration: none; }
    .topic-disclosure { border-top: 1px solid var(--line); }
    .topic-disclosure:first-of-type { border-top: 0; }
    .topic-disclosure > summary { display: flex; justify-content: space-between; gap: 1rem;
      align-items: center; padding: 1.1rem 0; cursor: pointer; font-weight: 800; list-style: none; }
    .topic-disclosure > summary::-webkit-details-marker { display: none; }
    .topic-disclosure > summary::after { content: "+"; display: grid; place-items: center;
      width: 1.75rem; height: 1.75rem; border-radius: 50%; background: var(--gray-soft);
      color: var(--blue); font-size: 1.2rem; }
    .topic-disclosure[open] > summary::after { content: "−"; }
    .topic-disclosure > summary h3 { margin: 0; font-size: 1rem; font-weight: 800; }
    .summary-meta { color: var(--muted); font-size: .82rem; font-weight: 650; }
    .topic-detail-body { padding: 0 0 1.5rem; }
    .stage-strip { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: .45rem;
      margin: .1rem 0 1.35rem; padding: 0; list-style: none; }
    .stage-strip li { min-width: 0; padding: .72rem .75rem; border: 1px solid var(--line);
      border-radius: .72rem; background: var(--surface-2); }
    .stage-strip strong { display: block; margin-top: .2rem; font-size: .76rem; line-height: 1.35;
      overflow-wrap: anywhere; }
    .stage-name { color: var(--blue); font-size: .68rem; font-weight: 850; letter-spacing: .055em;
      text-transform: uppercase; }
    .source-block { margin-top: 1.1rem; }
    .source-block h4 { margin: 0 0 .55rem; font-size: .8rem; color: var(--blue);
      letter-spacing: .08em; text-transform: uppercase; }
    .narrative { margin: 0; padding: 1rem 1.1rem; border-left: .28rem solid var(--blue);
      background: var(--surface-2); color: var(--ink); line-height: 1.68; }
    .subnarrative-disclosure { border: 1px solid var(--line); border-radius: .8rem;
      background: var(--surface-2); }
    .subnarrative-disclosure > summary { padding: .82rem 1rem; cursor: pointer; color: var(--blue);
      font-size: .84rem; font-weight: 850; }
    .subnarratives { display: grid; gap: .7rem; margin: 0; padding: .8rem; list-style: none;
      counter-reset: subnarrative; }
    .subnarrative { counter-increment: subnarrative; padding: .9rem 1rem; border: 1px solid var(--line);
      border-radius: .8rem; background: var(--surface); }
    .subnarrative-head { display: flex; gap: .65rem; align-items: flex-start; }
    .subnarrative-head::before { content: counter(subnarrative); flex: 0 0 auto; display: grid;
      place-items: center; width: 1.65rem; height: 1.65rem; border-radius: 50%;
      background: var(--blue-soft); color: var(--badge-blue); font-size: .78rem; font-weight: 900; }
    .subnarrative-text { margin: 0; line-height: 1.55; }
    .response { display: grid; gap: .55rem; }
    .response-item { display: grid; grid-template-columns: 2rem 1fr; gap: .7rem; padding: .75rem .85rem;
      border: 1px solid var(--line); border-radius: .75rem; background: var(--surface-2); }
    .response-number { color: var(--muted); font-size: .75rem; font-weight: 800; text-align: right;
      padding-top: .16rem; }
    .response-text { margin: 0; color: var(--ink); line-height: 1.62; overflow-wrap: anywhere; }
    .response-item.response-has-ns { border-color: #d96c6c; box-shadow: inset .24rem 0 0 var(--red); }
    .citations { display: inline-flex; flex-wrap: wrap; gap: .2rem; margin-left: .35rem;
      vertical-align: .08em; }
    .citation { padding: .14rem .36rem; border-radius: .35rem; background: var(--blue-soft);
      color: var(--badge-blue); font-size: .78rem; font-weight: 850; white-space: nowrap;
      display: inline-flex; align-items: center; gap: .24rem; border: 1px solid currentColor; }
    .citation-support { font-size: .72rem; letter-spacing: .045em; }
    .citation-unjudged { background: var(--gray-soft); color: var(--muted); }
    .inline-support-note { margin: 0 0 .65rem; color: var(--muted); font-size: .8rem; line-height: 1.55; }
    .mini-label { display: inline-block; margin: 0 .1rem; padding: .1rem .32rem; border-radius: .32rem;
      font-size: .74rem; font-weight: 900; }
    .privacy-note { margin: .75rem 0 0; color: var(--muted); font-size: .8rem; line-height: 1.5; }
    .judgment-summary { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .65rem; }
    .judgment-summary article { padding: .9rem 1rem; border: 1px solid #edc36e; border-radius: .8rem;
      background: var(--amber-soft); color: #6b3e00; }
    .judgment-summary article.judgment-verified { border-color: #82c99a; background: var(--green-soft);
      color: #14532d; }
    .judgment-summary strong { display: block; margin-top: .18rem; font-size: 1rem; }
    .judgment-summary p { margin: .32rem 0 0; font-size: .82rem; line-height: 1.45; }
    .judgment-label { font-size: .7rem; font-weight: 850; letter-spacing: .055em; text-transform: uppercase; }
    .support-metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .65rem;
      margin-top: .75rem; }
    .support-metrics article { padding: .9rem 1rem; border: 1px solid var(--line); border-radius: .8rem;
      background: var(--surface-2); }
    .support-metrics span { display: block; color: var(--muted); font-size: .7rem; font-weight: 800;
      letter-spacing: .045em; text-transform: uppercase; }
    .support-metrics strong { display: block; margin-top: .22rem; font-size: 1.2rem;
      font-variant-numeric: tabular-nums; }
    .judgment-tasks { margin-top: .75rem; border: 1px solid var(--line); border-radius: .8rem;
      background: var(--surface-2); }
    .judgment-tasks > summary { padding: .85rem 1rem; cursor: pointer; color: var(--blue);
      font-size: .85rem; font-weight: 800; }
    .judgment-task-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: .35rem .55rem; margin: 0; padding: 0 1rem 1rem; list-style: none; }
    .judgment-task { display: flex; justify-content: space-between; gap: .6rem; align-items: flex-start;
      min-width: 0; padding: .48rem .58rem; border: 1px solid var(--line); border-radius: .55rem;
      background: var(--surface); color: var(--muted); font-size: .8rem; }
    .judgment-claim { min-width: 0; font-weight: 700; }
    .judgment-statement { display: block; margin-top: .2rem; color: var(--muted); font-weight: 400;
      font-style: normal; line-height: 1.45; overflow-wrap: anywhere; }
    .judgment-statement::before { content: "“"; }
    .judgment-statement::after { content: "”"; }
    .judgment-badge { flex: 0 0 auto; padding: .18rem .4rem; border-radius: .38rem; font-size: .74rem;
      font-weight: 850; }
    .label-fs { background: var(--green-soft); color: var(--badge-green); }
    .label-ps { background: var(--amber-soft); color: var(--badge-amber); }
    .label-ns { background: var(--red-soft); color: var(--badge-red); }
    .label-none { background: var(--gray-soft); color: var(--muted); }
    .table-scroll { margin-top: 1rem; overflow-x: auto; border: 1px solid var(--line);
      border-radius: .85rem; background: var(--surface-2); }
    .metric-table { width: 100%; border-collapse: collapse; font-size: .82rem;
      font-variant-numeric: tabular-nums; }
    .metric-table caption { padding: .8rem 1rem .2rem; color: var(--muted); font-size: .78rem;
      font-weight: 800; letter-spacing: .045em; text-align: left; text-transform: uppercase; }
    .metric-table th, .metric-table td { padding: .55rem .8rem; text-align: right;
      border-top: 1px solid var(--line); white-space: nowrap; }
    .metric-table thead th { color: var(--muted); font-size: .74rem; font-weight: 800; }
    .metric-table th[scope="row"] { text-align: left; font-weight: 800; }
    .metric-table tbody tr:nth-child(odd) { background: var(--surface); }
    .metric-table .cell-na { color: var(--muted); font-weight: 700; }
    .checklist, .commands { display: grid; gap: .5rem; margin: 0; padding: 0; list-style: none; }
    .checklist li, .commands li { color: var(--muted); line-height: 1.45; }
    .commands { margin-top: .5rem; }
    code { padding: .12rem .32rem; border-radius: .32rem; background: var(--gray-soft);
      color: var(--ink); overflow-wrap: anywhere; }
    .provenance { display: flex; flex-wrap: wrap; gap: .6rem 1.2rem; margin-top: 1.25rem;
      padding-top: 1.1rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .86rem; }
    .visually-hidden { position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0;
      overflow: hidden; clip-path: inset(50%); white-space: nowrap; border: 0; }
    footer { padding: 1.5rem .25rem 0; color: var(--muted); font-size: .82rem; line-height: 1.55;
      text-align: center; }
    @media (max-width: 760px) {
      main { width: min(100% - 1rem, 70rem); padding-top: .5rem; }
      .hero, .section { border-radius: .95rem; }
      .stats, .metric-grid, .count-guide-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .availability-row { grid-template-columns: 1fr auto; gap: .45rem .8rem; }
      .availability-row p { grid-column: 1 / -1; }
      .topic-disclosure > summary { align-items: flex-start; }
      .summary-meta { display: block; margin-top: .2rem; }
      .response-item { grid-template-columns: 1.45rem 1fr; padding: .7rem .65rem; }
      .stage-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .judgment-summary, .judgment-task-list, .support-metrics { grid-template-columns: 1fr; }
    }
    @media (max-width: 410px) {
      .stats, .metric-grid, .count-guide-grid { grid-template-columns: 1fr; }
      h1 { font-size: 2.2rem; }
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0c1422; --surface: #141f31; --surface-2: #19263a; --ink: #edf4ff; --muted: #b6c4d9;
        --line: #2d3c54; --blue: #8bb8ff; --blue-soft: #1e3966; --green: #65d98b;
        --green-soft: #173d28; --amber: #ffd178; --amber-soft: #422e12; --red: #ff9b93;
        --red-soft: #4a201e; --gray-soft: #25334a; --shadow: none;
        --badge-blue: #8bb8ff; --badge-green: #65d98b; --badge-amber: #ffd178; --badge-red: #ff9b93;
      }
      .judgment-summary article { color: #ffe4ac; }
      .judgment-summary article.judgment-verified { color: #d5ffe1; }
      .hero-note { border-top-color: #6d5226; }
    }
    @media print {
      :root {
        --bg: #fff; --surface: #fff; --surface-2: #fff; --shadow: none; --ink: #111827;
        --muted: #374151; --line: #9ca3af; --blue: #1d4ed8; --blue-soft: #fff; --green: #166534;
        --green-soft: #fff; --amber: #7c4a02; --amber-soft: #fff; --red: #991b1b; --red-soft: #fff;
        --gray-soft: #fff; --badge-blue: #1d4ed8; --badge-green: #166534; --badge-amber: #7c4a02;
        --badge-red: #991b1b; color: var(--ink); background: #fff;
      }
      body { color: var(--ink); background: #fff; }
      main { width: 100%; padding: 0; }
      .hero, .section { break-inside: avoid; }
      .judgment-summary article, .judgment-summary article.judgment-verified, .hero-note {
        color: var(--ink); background: #fff; border-color: var(--line); }
      .citation, .judgment-badge, .mini-label, .status-na, .status-available {
        border: 1px solid var(--line); }
    }
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <meta name="robots" content="noindex,nofollow,noarchive">
  <title>TREC RAG private post-run evaluation</title>
  <style>{style}  </style>
</head>
<body>
  <main>
{body}
    <footer>Private post-run evaluation. It intentionally includes narratives, generated
    subnarratives, and answer text. It excludes corpus passages, document identifiers,
    credentials, and raw provider responses.</footer>
  </main>
</body>
</html>
"""
