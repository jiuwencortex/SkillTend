# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""JSONL experiment result logger.

All study runners write their results through this logger.  Each record is
one JSON object per line, enabling streaming reads and easy post-processing
with jq, pandas, or the study-specific analyze.py scripts.

Usage::

    logger = ResultLogger("results/study_02.jsonl", study="02")

    # Log one result
    logger.log(
        run_id="abc123",
        config={"skill_nudge_interval": 10, "memory_nudge_interval": 5},
        metrics={"q": 0.72, "rr": 0.81, "id_score": 0.44, "tsr": 0.70, "ps": 0.85},
        meta={"task_id": "code_debug_01", "session_length": 30},
    )

    # Read back all records
    records = ResultLogger.read("results/study_02.jsonl")
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class ResultLogger:
    """Thread-safe JSONL writer for experiment results.

    The file is opened in append mode so multiple processes can write to the
    same file if needed (each write is a single os.write call which is atomic
    on POSIX for small payloads).
    """

    def __init__(
        self,
        path: str | Path,
        study: str,
        run_group: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            path:        Output JSONL file path.  Parent directories are created.
            study:       Study identifier, e.g. "02".
            run_group:   Optional group label for this batch of runs (e.g. "sweep_a").
            extra_meta:  Key-value pairs added to every record written by this logger.
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._study = study
        self._run_group = run_group
        self._extra_meta = extra_meta or {}
        self._lock = threading.Lock()

    # ── Writing ───────────────────────────────────────────────────────────────

    def log(
        self,
        run_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one result record to the JSONL file.

        Args:
            run_id:  Unique identifier for this run (e.g. uuid4 hex or descriptive string).
            config:  Configuration that produced this result (e.g. ReviewerConfig fields).
            metrics: Measured outcomes (e.g. q, rr, tsr, tokens_used).
            meta:    Optional additional information (task_id, session_length, etc.).
        """
        record: Dict[str, Any] = {
            "study": self._study,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "metrics": metrics,
            "meta": {**self._extra_meta, **(meta or {})},
        }
        if self._run_group:
            record["run_group"] = self._run_group

        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def log_error(
        self,
        run_id: str,
        config: Dict[str, Any],
        error: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a failed run (no metrics)."""
        self.log(
            run_id=run_id,
            config=config,
            metrics={"error": error},
            meta={**(meta or {}), "_failed": True},
        )

    # ── Reading ───────────────────────────────────────────────────────────────

    @staticmethod
    def read(path: str | Path) -> List[Dict[str, Any]]:
        """Read all records from a JSONL file.  Returns an empty list if the file
        does not exist or contains no valid records."""
        p = Path(path)
        if not p.exists():
            return []
        records: List[Dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    @staticmethod
    def read_metrics(path: str | Path) -> List[Dict[str, Any]]:
        """Read only the metrics dicts (flattened with config and run_id)."""
        records = ResultLogger.read(path)
        result = []
        for r in records:
            if r.get("metrics", {}).get("error"):
                continue
            flat = {"run_id": r.get("run_id"), "study": r.get("study")}
            flat.update(r.get("config", {}))
            flat.update(r.get("metrics", {}))
            flat.update(r.get("meta", {}))
            result.append(flat)
        return result

    @staticmethod
    def stream(path: str | Path) -> Iterable[Dict[str, Any]]:
        """Stream records from a JSONL file one at a time (memory-efficient for large files)."""
        p = Path(path)
        if not p.exists():
            return
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the number of records in the output file so far."""
        if not self._path.exists():
            return 0
        return sum(1 for line in self._path.read_text(encoding="utf-8").splitlines() if line.strip())

    def __repr__(self) -> str:
        return f"ResultLogger(study={self._study!r}, path={self._path}, records={self.count()})"
