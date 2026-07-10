"""SFCHD dataset tooling: split parsing, image/label reconciliation, YOLO config.

The published SFCHD class schema is used unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CLASS_NAMES: tuple[str, ...] = (
    "person",
    "helmet",
    "self_clothes",
    "safety_clothes",
    "head",
    "blur_head",
    "blur_clothes",
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_split_file(path: Path) -> list[str]:
    """Return image basenames (without extension) from an upstream split file.

    Upstream split files list absolute paths from the authors' machine, so only
    the basename carries information.
    """
    stems = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            stems.append(Path(line).stem)
    return stems


@dataclass
class ReconcileReport:
    """Result of joining on-disk images against on-disk label files by stem."""

    matched: list[str] = field(default_factory=list)
    images_without_labels: list[str] = field(default_factory=list)
    labels_without_images: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.matched) + len(self.images_without_labels) + len(self.labels_without_images)

    @property
    def match_rate(self) -> float:
        return len(self.matched) / self.total if self.total else 0.0

    @property
    def orphan_rate(self) -> float:
        return 1.0 - self.match_rate

    def summary(self) -> str:
        return (
            f"matched: {len(self.matched)}\n"
            f"images without labels: {len(self.images_without_labels)}\n"
            f"labels without images: {len(self.labels_without_images)}\n"
            f"match rate: {self.match_rate:.4%}"
        )


def reconcile(image_stems: set[str], label_stems: set[str]) -> ReconcileReport:
    return ReconcileReport(
        matched=sorted(image_stems & label_stems),
        images_without_labels=sorted(image_stems - label_stems),
        labels_without_images=sorted(label_stems - image_stems),
    )


def collect_image_stems(images_dir: Path) -> dict[str, str]:
    """Map stem -> filename for every image in the directory."""
    stems: dict[str, str] = {}
    for entry in images_dir.iterdir():
        if entry.suffix.lower() in IMAGE_EXTENSIONS:
            stems[entry.stem] = entry.name
    return stems


def redundant_splits(split_members: dict[str, set[str]]) -> list[str]:
    """Names of splits whose members are fully contained in another split.

    The upstream SFCHD test list is a 6-image subset of val — a leftover, not a
    real split. Keeping it would imply an evaluation set that doesn't exist.
    """
    redundant = []
    for name, members in split_members.items():
        if not members:
            continue
        for other_name, other_members in split_members.items():
            if other_name != name and members <= other_members:
                redundant.append(name)
                break
    return redundant


def write_split_list(path: Path, images_dir: Path, filenames: list[str]) -> None:
    path.write_text("".join(f"{(images_dir / name).resolve()}\n" for name in sorted(filenames)))


def write_dataset_yaml(path: Path, dataset_root: Path, splits: dict[str, str]) -> None:
    """Write an ultralytics dataset config referencing split list files."""
    lines = [f"path: {dataset_root.resolve()}"]
    for split, list_file in splits.items():
        lines.append(f"{split}: {list_file}")
    lines.append("names:")
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    path.write_text("\n".join(lines) + "\n")
