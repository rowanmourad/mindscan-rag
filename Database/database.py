"""
database.py - SQL Server connectivity for the Brain Tumor Detection API.

Design notes
------------
We do NOT open one global connection/cursor at import time. FastAPI serves
requests concurrently, and a single pyodbc connection/cursor is not safe to
share across concurrent requests. Instead, each request gets its own
connection through the `get_db` FastAPI dependency below; pyodbc's built-in
connection pooling (enabled by default) keeps this cheap.

Usage in app.py:

    from fastapi import Depends
    import pyodbc
    from database import get_db

    @app.get("/patients")
    def get_patients(db: pyodbc.Cursor = Depends(get_db)):
        db.execute("SELECT PatientID, FullName, Age, Gender FROM Patient")
        rows = db.fetchall()
        return [dict(zip([c[0] for c in db.description], row)) for row in rows]
"""
from __future__ import annotations

from typing import Iterator

import pyodbc

from config import settings


def get_connection() -> pyodbc.Connection:
    """Open a new SQL Server connection using the configured connection string."""
    return pyodbc.connect(settings.connection_string)


def get_db() -> Iterator[pyodbc.Cursor]:
    """
    FastAPI dependency. Yields a cursor scoped to a single request.
    Commits on success, rolls back on exception, always closes the connection.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def check_connection() -> bool:
    """Used by /health to confirm the DB is reachable, without raising."""
    try:
        conn = get_connection()
        conn.close()
        return True
    except Exception:
        return False


def row_to_dict(cursor: pyodbc.Cursor, row) -> dict:
    """Helper: convert a pyodbc row into a plain dict using cursor column names."""
    return dict(zip([col[0] for col in cursor.description], row))
