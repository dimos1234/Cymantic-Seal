"""SQLite database for seal history (no payments)."""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "cymatic_seal.db"

HMAC_SECRET = os.environ.get("CYMATIC_HMAC_SECRET", "cymatic-seal-default-secret-change-me")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seal_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL,
            identifier TEXT NOT NULL,
            credits_used INTEGER NOT NULL DEFAULT 0,
            original_filename TEXT NOT NULL DEFAULT '',
            duration_seconds REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def hash_identifier(ip: str) -> str:
    return hmac.new(
        HMAC_SECRET.encode(), ip.encode(), hashlib.sha256
    ).hexdigest()[:32]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_seal_job(
    job_id: str,
    identifier: str,
    original_filename: str = "",
    duration_seconds: float = 0.0,
) -> None:
    conn = _connect()
    conn.execute(
        "INSERT OR IGNORE INTO seal_jobs (job_id, identifier, credits_used, original_filename, duration_seconds, created_at) "
        "VALUES (?, ?, 0, ?, ?, ?)",
        (job_id, identifier, original_filename, duration_seconds, _now_iso()),
    )
    conn.commit()
    conn.close()


def get_seal_history(identifier: str, limit: int = 50) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT job_id, original_filename, duration_seconds, created_at "
        "FROM seal_jobs WHERE identifier = ? ORDER BY created_at DESC LIMIT ?",
        (identifier, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
