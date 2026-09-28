"""Decide whether a CI run only changed documentation files."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import PurePosixPath


DOCUMENT_ASSETS = {".gif", ".ico", ".jpeg", ".jpg", ".mp3", ".mp4", ".pdf", ".png", ".svg", ".webp"}


def is_docs_only(paths: list[str]) -> bool:
    """Return true only for a nonempty set of documentation and static assets."""
    return bool(paths) and all(
        path.endswith(".md")
        or path == "zensical.toml"
        or (
            path.startswith(("docs/", "assets/readme/"))
            and PurePosixPath(path).suffix.lower() in DOCUMENT_ASSETS
        )
        for path in paths
    )


def changed_paths(base: str, head: str) -> list[str]:
    if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (base, head)):
        raise ValueError("CI needs two commit SHAs to inspect changed paths")
    if base == "0" * 40:
        return []  # A new branch has no earlier commit; run the full suite.
    output = subprocess.check_output(
        ["git", "diff", "--name-only", "-z", "--no-renames", base, head]
    )
    return [os.fsdecode(path) for path in output.split(b"\0") if path]


if __name__ == "__main__":
    paths = changed_paths(os.environ["BASE_SHA"], os.environ["HEAD_SHA"])
    docs_only = str(is_docs_only(paths)).lower()
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"docs_only={docs_only}\n")
    print(f"Changed files: {len(paths)}; documentation only: {docs_only}")
