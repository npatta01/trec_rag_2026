from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "code/tools/run_headless_chrome.py"


def _fake_browser(tmp_path: Path) -> Path:
    browser = tmp_path / "fake-chrome"
    browser.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            from pathlib import Path
            import sys

            record = {
                "args": sys.argv[1:],
                "TMPDIR": os.environ.get("TMPDIR"),
                "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
            }
            Path(os.environ["FAKE_BROWSER_RECORD"]).write_text(json.dumps(record))
            screenshot = next(
                value.split("=", 1)[1]
                for value in sys.argv[1:]
                if value.startswith("--screenshot=")
            )
            if os.environ.get("FAKE_BROWSER_EMPTY") == "1":
                Path(screenshot).touch()
            else:
                Path(screenshot).write_bytes(b"fake-png")
            raise SystemExit(int(os.environ.get("FAKE_BROWSER_EXIT", "0")))
            """
        ),
        encoding="utf-8",
    )
    browser.chmod(0o755)
    return browser


def _run_wrapper(
    tmp_path: Path,
    *,
    extra_env: dict[str, str] | None = None,
    url: str = "file:///absolute/report.html",
) -> tuple[subprocess.CompletedProcess[str], dict[str, object], Path, Path]:
    browser = _fake_browser(tmp_path)
    record_path = tmp_path / "browser-record.json"
    output_path = tmp_path / "report.png"
    scratch_root = tmp_path / "browser-qa"
    env = {**os.environ, "FAKE_BROWSER_RECORD": str(record_path), **(extra_env or {})}
    completed = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--browser",
            str(browser),
            "--scratch-root",
            str(scratch_root),
            f"--url={url}",
            "--output",
            str(output_path),
            "--width",
            "1440",
            "--height",
            "1100",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    return completed, record, output_path, scratch_root


def test_wrapper_rejects_option_shaped_url_before_starting_browser(tmp_path: Path) -> None:
    completed, record, output_path, scratch_root = _run_wrapper(
        tmp_path,
        url="--user-data-dir=/tmp/attacker-profile",
    )

    assert completed.returncode != 0
    assert "URL scheme" in completed.stderr
    assert record == {}
    assert not output_path.exists()
    assert not scratch_root.exists()


@pytest.mark.parametrize(
    "url",
    [
        "file:relative-report.html",
        "https:///missing-host/report.html",
        "ftp://example.test/report.html",
    ],
)
def test_wrapper_rejects_malformed_or_unsupported_url(
    tmp_path: Path,
    url: str,
) -> None:
    completed, record, output_path, scratch_root = _run_wrapper(tmp_path, url=url)

    assert completed.returncode != 0
    assert "URL" in completed.stderr
    assert record == {}
    assert not output_path.exists()
    assert not scratch_root.exists()


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/report.html",
        "https://example.test/report.html",
    ],
)
def test_wrapper_accepts_http_and_https_urls(tmp_path: Path, url: str) -> None:
    completed, record, output_path, scratch_root = _run_wrapper(tmp_path, url=url)

    assert completed.returncode == 0, completed.stderr
    assert record["args"][-1] == url
    assert output_path.read_bytes() == b"fake-png"
    assert scratch_root.is_dir()
    assert list(scratch_root.iterdir()) == []


def test_wrapper_places_environment_profile_and_cache_in_private_root(tmp_path: Path) -> None:
    completed, record, output_path, scratch_root = _run_wrapper(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert output_path.read_bytes() == b"fake-png"
    args = record["args"]
    assert "--headless=new" in args
    assert "--no-sandbox" in args
    assert "--disable-gpu" in args
    assert "--disable-dev-shm-usage" in args
    assert "--window-size=1440,1100" in args
    assert "file:///absolute/report.html" in args

    private_paths = [
        Path(record["TMPDIR"]),
        Path(record["XDG_CACHE_HOME"]),
        Path(next(value.split("=", 1)[1] for value in args if value.startswith("--user-data-dir="))),
        Path(next(value.split("=", 1)[1] for value in args if value.startswith("--disk-cache-dir="))),
    ]
    private_root = private_paths[0].parent
    assert private_root.parent == scratch_root
    assert all(path.is_relative_to(private_root) for path in private_paths)
    assert scratch_root.is_dir()
    assert list(scratch_root.iterdir()) == []


def test_wrapper_cleans_private_root_when_browser_fails(tmp_path: Path) -> None:
    completed, record, _, scratch_root = _run_wrapper(
        tmp_path,
        extra_env={"FAKE_BROWSER_EXIT": "9"},
    )

    assert completed.returncode != 0
    assert record
    assert scratch_root.is_dir()
    assert list(scratch_root.iterdir()) == []


def test_wrapper_rejects_empty_screenshot_and_cleans_private_root(tmp_path: Path) -> None:
    completed, record, output_path, scratch_root = _run_wrapper(
        tmp_path,
        extra_env={"FAKE_BROWSER_EMPTY": "1"},
    )

    assert completed.returncode != 0
    assert record
    assert output_path.exists()
    assert output_path.stat().st_size == 0
    assert scratch_root.is_dir()
    assert list(scratch_root.iterdir()) == []
