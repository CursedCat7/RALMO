"""RALMO request logger.

Logs per-request metrics to CSV or SQLite for analysis and reproducibility.
Captures prompt, token counts, acceptance rate, latency, and energy estimates.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ralmo_core.speculative import SpeculativeResult

logger = logging.getLogger(__name__)


class RALMOLogger:
    """Per-request metrics logger for RALMO.

    Supports CSV and optional SQLite output. Each request is logged as a single
    row with columns for prompt, token counts, acceptance statistics, latency,
    and energy consumption.

    Attributes:
        output_dir: Directory for log files.
        log_format: Output format ('csv' or 'sqlite').
        verbose: Whether to also print to console.
    """

    CSV_COLUMNS = [
        "timestamp",
        "prompt_preview",
        "total_tokens",
        "draft_proposed",
        "draft_accepted",
        "draft_rejected",
        "target_corrections",
        "acceptance_rate",
        "iterations",
        "latency_ms",
        "energy_joules",
        "finish_reason",
    ]

    def __init__(
        self,
        output_dir: str = "./logs",
        log_format: str = "csv",
        verbose: bool = False,
    ) -> None:
        """Initialize the RALMO logger.

        Args:
            output_dir: Directory to write log files.
            log_format: 'csv' or 'sqlite'.
            verbose: Print logs to console as well.
        """
        self.output_dir = Path(output_dir)
        self.log_format = log_format
        self.verbose = verbose
        self._csv_writer: csv.DictWriter | None = None
        self._csv_file: Any = None
        self._sqlite_conn: sqlite3.Connection | None = None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if log_format == "csv":
            self._init_csv()
        elif log_format == "sqlite":
            self._init_sqlite()

    def _init_csv(self) -> None:
        """Initialize CSV writer."""
        csv_path = self.output_dir / "ralmo_log.csv"
        file_exists = csv_path.exists()
        self._csv_file = open(csv_path, "a", newline="", encoding="utf-8")  # noqa: SIM115
        # Type ignore is needed here because Python 3.11 DictWriter is typed differently in stubs
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_COLUMNS)  # type: ignore[arg-type]
        if not file_exists:
            self._csv_writer.writeheader()
        logger.info("CSV logger initialized: %s", csv_path)

    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        db_path = self.output_dir / "ralmo_log.db"
        self._sqlite_conn = sqlite3.connect(str(db_path))
        self._sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prompt_preview TEXT,
                total_tokens INTEGER,
                draft_proposed INTEGER,
                draft_accepted INTEGER,
                draft_rejected INTEGER,
                target_corrections INTEGER,
                acceptance_rate REAL,
                iterations INTEGER,
                latency_ms REAL,
                energy_joules REAL,
                finish_reason TEXT
            )
        """)
        self._sqlite_conn.commit()
        logger.info("SQLite logger initialized: %s", db_path)

    def log_request(
        self,
        prompt: str,
        result: SpeculativeResult,
        energy_joules: float = 0.0,
    ) -> None:
        """Log a single inference request.

        Args:
            prompt: The input prompt text.
            result: The speculative decoding result with stats.
            energy_joules: Estimated energy consumption.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        prompt_preview = prompt[:80].replace("\n", " ")
        stats = result.stats

        row = {
            "timestamp": timestamp,
            "prompt_preview": prompt_preview,
            "total_tokens": stats.total_tokens,
            "draft_proposed": stats.draft_tokens_proposed,
            "draft_accepted": stats.draft_tokens_accepted,
            "draft_rejected": stats.draft_tokens_rejected,
            "target_corrections": stats.target_corrections,
            "acceptance_rate": round(stats.acceptance_rate, 4),
            "iterations": stats.iterations,
            "latency_ms": round(stats.latency_ms, 2),
            "energy_joules": round(energy_joules, 6),
            "finish_reason": result.finish_reason,
        }

        if self.log_format == "csv" and self._csv_writer is not None:
            self._csv_writer.writerow(row)
            if self._csv_file is not None:
                self._csv_file.flush()
        elif self.log_format == "sqlite" and self._sqlite_conn is not None:
            cols = ", ".join(row.keys())
            placeholders = ", ".join(["?"] * len(row))
            self._sqlite_conn.execute(
                f"INSERT INTO requests ({cols}) VALUES ({placeholders})",
                list(row.values()),
            )
            self._sqlite_conn.commit()

        if self.verbose:
            logger.info(
                "Request logged: %d tokens, %.1fms, %.1f%% acceptance",
                stats.total_tokens,
                stats.latency_ms,
                stats.acceptance_rate * 100,
            )

    def close(self) -> None:
        """Close log handles."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        if self._sqlite_conn is not None:
            self._sqlite_conn.close()
            self._sqlite_conn = None
        logger.info("Logger closed.")
