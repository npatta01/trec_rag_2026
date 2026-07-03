"""Repository-local environment file loading."""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "AGENTS.md").exists():
            return candidate
    return current


def shared_checkout_root(repo_root: Path) -> Path | None:
    git_file = repo_root / ".git"
    if not git_file.is_file():
        return None
    git_pointer = git_file.read_text(encoding="utf-8").strip()
    if not git_pointer.startswith("gitdir:"):
        return None
    git_dir = Path(git_pointer.split(":", 1)[1].strip()).expanduser()
    if not git_dir.is_absolute():
        git_dir = (repo_root / git_dir).resolve()
    if git_dir.parent.name != "worktrees":
        return None
    return git_dir.parent.parent.parent


def load_dotenv(
    path: Path,
    *,
    override: bool = False,
    protected_keys: set[str] | None = None,
) -> None:
    if not path.exists():
        return
    protected_keys = protected_keys or set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if override and key not in protected_keys:
            os.environ[key] = value.strip().strip('"').strip("'")
        else:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def load_repo_env(repo_root: Path) -> None:
    protected_keys = set(os.environ)
    roots = []
    shared_root = shared_checkout_root(repo_root)
    if shared_root and shared_root != repo_root:
        roots.append(shared_root)
    roots.append(repo_root)
    for root in roots:
        load_dotenv(root / ".env", override=True, protected_keys=protected_keys)
        load_dotenv(root / ".env.local", override=True, protected_keys=protected_keys)
