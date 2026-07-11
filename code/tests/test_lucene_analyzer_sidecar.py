import json
import os

import pytest

from trec_rag.query_analyzer import (
    AnalyzerFingerprint,
    RemoteLuceneQueryAnalyzer,
    stable_unique,
)


FINGERPRINT = {
    "contract_version": "lucene_default_english_v1",
    "implementation": "local_lucene_reference_server_v1",
    "lucene_version": "10.4.0",
    "analyzer_class": "io.anserini.analysis.DefaultEnglishAnalyzer chain",
    "tokenizer": "org.apache.lucene.analysis.standard.StandardTokenizer",
    "filters": [
        "EnglishPossessiveFilter",
        "LowerCaseFilter",
        "StopFilter(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET)",
        "PorterStemFilter",
    ],
    "stopword_sha256": "2f66c0e3dde5d31c7e919e2ed4d9d91390696480be361bfa143ca9ae0cb7ca13",
    "unicode_version": "Lucene-10.4.0-StandardTokenizer-UAX29",
    "index_id": "hosted_climbmix_unknown_revision",
}


def test_analyzer_fingerprint_and_stable_unique_contract():
    fingerprint = AnalyzerFingerprint.from_mapping(FINGERPRINT)

    assert fingerprint.lucene_version == "10.4.0"
    assert fingerprint.filters[-1] == "PorterStemFilter"
    assert stable_unique(("bank", "bank", "cost")) == ("bank", "cost")
    with pytest.raises(ValueError, match="filters"):
        AnalyzerFingerprint.from_mapping({**FINGERPRINT, "filters": "not-an-array"})


@pytest.mark.skipif(
    os.getenv("RUN_LUCENE_ANALYZER_INTEGRATION") != "1",
    reason="requires the pinned localhost Lucene analyzer sidecar",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("City buses are running on time.", ("citi", "buse", "run", "time")),
        (
            "open-source banks banking bank's not how",
            ("open", "sourc", "bank", "bank", "bank", "how"),
        ),
        ("no not such then there these will", ()),
        ("Ｆｕｌｌｗｉｄｔｈ café cafe\u0301", ("ｆｕｌｌｗｉｄｔｈ", "café", "cafe\u0301")),
        ("中文测试 日本語テスト", ("中", "文", "测", "试", "日", "本", "語", "テスト")),
        ("rocket🚀launch smile🙂face", ("rocket", "🚀", "launch", "smile", "🙂", "face")),
        ("U.S.A. NASA e.g. v2.0 ２０２６ 3.14", ("u.s.a", "nasa", "e.g", "v2.0", "２０２６", "3.14")),
        ("running running runs runner", ("run", "run", "run", "runner")),
        ("O’Reilly's co-op—costs €20?", ("o’reilli", "co", "op", "cost", "20")),
    ],
)
def test_pinned_lucene_sidecar_conformance_corpus(text, expected):
    analyzer = RemoteLuceneQueryAnalyzer()
    analyzed = analyzer.analyze(text)

    assert analyzed.tokens == expected
    assert analyzed.unique_tokens == stable_unique(expected)
    assert analyzed.fingerprint == AnalyzerFingerprint.from_mapping(FINGERPRINT)


def test_remote_client_rejects_a_changed_fingerprint(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    expected = AnalyzerFingerprint.from_mapping(FINGERPRINT)
    changed = {**FINGERPRINT, "lucene_version": "future"}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda _request, timeout: Response(
            {"tokens": ["bank"], "fingerprint": changed}
        ),
    )
    analyzer = RemoteLuceneQueryAnalyzer(expected_fingerprint=expected)

    with pytest.raises(ValueError, match="fingerprint"):
        analyzer.analyze("banks")
