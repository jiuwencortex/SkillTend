# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Interactive CLI for human labelling of skill quality.

Usage::

    python -m phase_1_foundations.study_01_skill_quality_metrics.label_tool \
        --skills-dir /path/to/skills \
        --output results/labels.jsonl \
        --rater-id alice

Displays each SKILL.md file and its associated task, asks the rater to assign
a quality label (1=low, 2=medium, 3=high) and optionally a short justification.
Saves labels to a JSONL file for use by compute_validity.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _collect_skill_paths(skills_dir: Path) -> list[Path]:
    return sorted(skills_dir.rglob("SKILL.md"))


def _display_skill(skill_path: Path, index: int, total: int) -> None:
    print("\n" + "=" * 72)
    print(f"  Skill {index + 1}/{total}: {skill_path.parent.name}")
    print("=" * 72)
    content = skill_path.read_text(encoding="utf-8")
    # Print at most 80 lines to keep display manageable
    lines = content.splitlines()[:80]
    print("\n".join(lines))
    if len(content.splitlines()) > 80:
        print(f"\n  ... ({len(content.splitlines()) - 80} more lines)")


def _ask_rating(skill_name: str) -> tuple[int, str]:
    """Prompt the rater for a quality label and justification."""
    while True:
        raw = input("\nQuality label [1=low, 2=medium, 3=high, s=skip, q=quit]: ").strip().lower()
        if raw == "q":
            sys.exit(0)
        if raw == "s":
            return -1, ""
        if raw in ("1", "2", "3"):
            label = int(raw)
            justification = input("Short justification (optional, press Enter to skip): ").strip()
            return label, justification
        print("  Please enter 1, 2, 3, s (skip), or q (quit).")


def run_label_tool(skills_dir: Path, output_path: Path, rater_id: str) -> None:
    skill_paths = _collect_skill_paths(skills_dir)
    if not skill_paths:
        print(f"No SKILL.md files found in {skills_dir}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-labelled skills to allow resuming
    labelled: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("rater_id") == rater_id:
                    labelled.add(rec["skill_name"])
            except (json.JSONDecodeError, KeyError):
                pass

    remaining = [p for p in skill_paths if p.parent.name not in labelled]
    print(f"\nSkillTend Study 01 — Human Labelling Tool")
    print(f"Rater: {rater_id}")
    print(f"Skills directory: {skills_dir}")
    print(f"Output: {output_path}")
    print(f"Total skills: {len(skill_paths)}  |  Already labelled: {len(labelled)}  |  Remaining: {len(remaining)}")

    with output_path.open("a", encoding="utf-8") as f:
        for i, skill_path in enumerate(remaining):
            _display_skill(skill_path, i, len(remaining))
            label, justification = _ask_rating(skill_path.parent.name)
            if label == -1:
                print("  Skipped.")
                continue
            record = {
                "rater_id": rater_id,
                "skill_name": skill_path.parent.name,
                "skill_path": str(skill_path),
                "label": label,
                "label_text": {1: "low", 2: "medium", 3: "high"}[label],
                "justification": justification,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"  Saved: {label}/3")

    print(f"\nDone. Labels written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 01: Human skill quality labelling")
    parser.add_argument("--skills-dir", required=True, type=Path, help="Directory containing SKILL.md files")
    parser.add_argument("--output", default="results/study_01_labels.jsonl", type=Path)
    parser.add_argument("--rater-id", required=True, help="Unique identifier for this rater")
    args = parser.parse_args()
    run_label_tool(args.skills_dir, args.output, args.rater_id)


if __name__ == "__main__":
    main()
