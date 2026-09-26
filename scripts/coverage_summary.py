"""Show line and branch coverage separately in CI."""

import json
import os
from pathlib import Path


totals = json.loads(Path("coverage.json").read_text())["totals"]
line = totals["percent_statements_covered"]
branch = totals["percent_branches_covered"]
summary = f"Line coverage: {line:.2f}%\nBranch coverage: {branch:.2f}%"
print(summary)
if destination := os.environ.get("GITHUB_STEP_SUMMARY"):
    with Path(destination).open("a") as file:
        file.write(f"## Coverage\n\nLine: {line:.2f}%\n\nBranch: {branch:.2f}%\n")
if totals["missing_lines"] or totals["missing_branches"]:
    raise SystemExit("Line and branch coverage must both be 100%")
