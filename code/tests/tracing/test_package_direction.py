import tomllib
from pathlib import Path


def test_reusable_tracing_does_not_import_experiments():
    tracing_root = Path(__file__).parents[2] / "trec_rag" / "tracing"
    sources = "\n".join(path.read_text() for path in tracing_root.glob("*.py"))
    assert "trec_rag.experiments" not in sources


def test_reusable_trace_models_are_available_from_the_tracing_package():
    from trec_rag.tracing.models import SpanSpec, TraceBundle

    assert SpanSpec.__name__ == "SpanSpec"
    assert TraceBundle.__name__ == "TraceBundle"


def test_reusable_trace_export_is_available_from_the_tracing_package():
    from trec_rag.tracing.phoenix_export import export_trace_bundle

    assert export_trace_bundle.__name__ == "export_trace_bundle"


def test_openai_semantics_dependency_is_declared_for_core_installs():
    project_root = Path(__file__).parents[3]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text())

    assert any(
        dependency.startswith("openinference-semantic-conventions")
        for dependency in metadata["project"]["dependencies"]
    )
