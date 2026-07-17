"""Capture a page with Chrome using disk-backed private browser state."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit


DEFAULT_SCRATCH_ROOT = Path.home() / ".cache/trec-rag/browser-qa"


def _browser_executable(browser: str | None) -> str:
    executable = shutil.which(browser or "google-chrome")
    if executable is None:
        name = browser or "google-chrome"
        raise FileNotFoundError(f"Chrome executable not found: {name}")
    return executable


def _validated_url(url: str) -> str:
    if any(ord(character) < 32 or ord(character) == 127 for character in url):
        raise ValueError("URL contains a control character")
    parsed = urlsplit(url)
    if parsed.scheme not in {"file", "http", "https"}:
        raise ValueError("URL scheme must be file, http, or https")
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"} or not parsed.path.startswith("/"):
            raise ValueError("file URL must use an absolute local path")
    elif parsed.hostname is None:
        raise ValueError("http/https URL must include a host")
    return url


def capture_page(
    *,
    url: str,
    output: Path,
    width: int,
    height: int,
    browser: str | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> Path:
    """Capture ``url`` while keeping all Chrome state below ``scratch_root``."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    url = _validated_url(url)
    executable = _browser_executable(browser)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    scratch_root = Path(scratch_root).expanduser().resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="chrome-", dir=scratch_root))
    tmp_dir = private_root / "tmp"
    xdg_cache = private_root / "xdg-cache"
    profile_dir = private_root / "profile"
    disk_cache = private_root / "disk-cache"

    try:
        for directory in (tmp_dir, xdg_cache, profile_dir, disk_cache):
            directory.mkdir()
        environment = {
            **os.environ,
            "TMPDIR": str(tmp_dir),
            "XDG_CACHE_HOME": str(xdg_cache),
        }
        command = [
            executable,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            f"--user-data-dir={profile_dir}",
            f"--disk-cache-dir={disk_cache}",
            f"--window-size={width},{height}",
            f"--screenshot={output}",
            url,
        ]
        subprocess.run(command, check=True, env=environment)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Chrome did not create a nonempty screenshot: {output}")
        return output
    finally:
        shutil.rmtree(private_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--browser")
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    args = parser.parse_args(argv)
    capture_page(
        url=args.url,
        output=args.output,
        width=args.width,
        height=args.height,
        browser=args.browser,
        scratch_root=args.scratch_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
