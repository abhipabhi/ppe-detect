#!/usr/bin/env python
"""Inject result tables from results/metrics.md into README.md.

README result tables are never hand-typed: this script copies them from the
generated results/metrics.md into marked blocks. Run after scripts/evaluate.py:

  python scripts/sync_readme.py

Markers in README.md:
  <!-- table:standard:start --> ... <!-- table:standard:end -->
  <!-- table:ab:start -->       ... <!-- table:ab:end -->
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def extract_table(markdown: str, heading_prefix: str) -> str:
    """Return the first markdown table that follows the heading starting with prefix."""
    lines = markdown.splitlines()
    in_section = False
    table: list[str] = []
    for line in lines:
        if line.startswith("#") and heading_prefix in line:
            in_section = True
            continue
        if in_section:
            if line.startswith("|"):
                table.append(line)
            elif table:
                break
            elif line.startswith("#"):
                raise ValueError(f"no table found under heading containing {heading_prefix!r}")
    if not table:
        raise ValueError(f"no table found under heading containing {heading_prefix!r}")
    return "\n".join(table)


def replace_marked_block(text: str, marker: str, replacement: str) -> str:
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    if start not in text or end not in text:
        raise ValueError(f"markers for {marker!r} not found")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{replacement}\n{end}{tail}"


def main() -> int:
    metrics = (REPO_ROOT / "results" / "metrics.md").read_text()
    readme_path = REPO_ROOT / "README.md"
    readme = readme_path.read_text()

    readme = replace_marked_block(
        readme, "table:standard", extract_table(metrics, "Standard evaluation")
    )
    readme = replace_marked_block(readme, "table:ab", extract_table(metrics, "Low-light A/B"))

    readme_path.write_text(readme)
    print("README tables synced from results/metrics.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
