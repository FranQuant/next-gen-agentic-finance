from __future__ import annotations

import json
import sqlite3
from pathlib import Path

try:
    from ..schemas import ResearchPacket, RunRecord
except ImportError:  # pragma: no cover
    from schemas import ResearchPacket, RunRecord


class SQLiteStorage:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    query TEXT NOT NULL,
                    stance_signal TEXT NOT NULL,
                    portfolio_signal TEXT,
                    conviction REAL NOT NULL,
                    report TEXT NOT NULL,
                    packet_json TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
            if "stance_signal" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN stance_signal TEXT")
            if "portfolio_signal" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN portfolio_signal TEXT")
            conn.execute(
                """
                UPDATE runs
                SET stance_signal = COALESCE(stance_signal, portfolio_signal)
                WHERE stance_signal IS NULL OR stance_signal = ''
                """
            )
            conn.commit()

    def save_run(self, packet: ResearchPacket) -> None:
        payload = json.dumps(packet.to_dict(), ensure_ascii=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id,
                    created_at,
                    query,
                    stance_signal,
                    portfolio_signal,
                    conviction,
                    report,
                    packet_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    packet.run_id,
                    packet.created_at,
                    packet.query,
                    packet.tactical_view.signal,
                    packet.tactical_view.signal,
                    packet.tactical_view.conviction,
                    packet.report,
                    payload,
                ),
            )
            conn.commit()

    def get_recent_runs(self, limit: int = 5) -> list[RunRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT run_id, created_at, query, COALESCE(stance_signal, portfolio_signal), conviction, report
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            RunRecord(
                run_id=row[0],
                created_at=row[1],
                query=row[2],
                stance_signal=row[3],
                conviction=float(row[4]),
                notes=(row[5].splitlines()[0] if row[5] else ""),
            )
            for row in rows
        ]
