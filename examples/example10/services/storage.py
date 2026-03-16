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
                    portfolio_signal TEXT NOT NULL,
                    conviction REAL NOT NULL,
                    report TEXT NOT NULL,
                    packet_json TEXT NOT NULL
                )
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
                    portfolio_signal,
                    conviction,
                    report,
                    packet_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    packet.run_id,
                    packet.created_at,
                    packet.query,
                    packet.portfolio_view.signal,
                    packet.portfolio_view.conviction,
                    packet.report,
                    payload,
                ),
            )
            conn.commit()

    def get_recent_runs(self, limit: int = 5) -> list[RunRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT run_id, created_at, query, portfolio_signal, conviction, report
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
                portfolio_signal=row[3],
                conviction=float(row[4]),
                notes=(row[5].splitlines()[0] if row[5] else ""),
            )
            for row in rows
        ]
