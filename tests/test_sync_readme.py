import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "sync_readme", Path(__file__).resolve().parents[1] / "scripts" / "sync_readme.py"
)
sync_readme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_readme)

METRICS = """# Evaluation results

## Standard evaluation — SFCHD val split

| Class | mAP50 |
|---|---|
| all | 0.746 |

## Low-light A/B — synthetically darkened val split (pre-registered)

Some prose.

| Class | Δ |
|---|---|
| all | +0.004 |
"""


def test_extract_table_by_heading():
    table = sync_readme.extract_table(METRICS, "Standard evaluation")
    assert table.splitlines()[0] == "| Class | mAP50 |"
    assert "| all | 0.746 |" in table


def test_extract_table_skips_prose_between_heading_and_table():
    table = sync_readme.extract_table(METRICS, "Low-light A/B")
    assert "| all | +0.004 |" in table


def test_extract_table_missing_heading_raises():
    with pytest.raises(ValueError):
        sync_readme.extract_table(METRICS, "Nonexistent section")


def test_replace_marked_block_roundtrip():
    text = "before\n<!-- table:x:start -->\nOLD\n<!-- table:x:end -->\nafter"
    out = sync_readme.replace_marked_block(text, "table:x", "NEW")
    assert "OLD" not in out
    assert "NEW" in out
    assert out.startswith("before\n")
    assert out.endswith("\nafter")
    # idempotent: markers survive so a second sync works
    again = sync_readme.replace_marked_block(out, "table:x", "NEWER")
    assert "NEWER" in again and "NEW\n" not in again


def test_replace_marked_block_missing_markers_raises():
    with pytest.raises(ValueError):
        sync_readme.replace_marked_block("no markers here", "table:x", "NEW")
