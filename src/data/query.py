"""DuckDB queries against the materialized experiment."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.utils.config import DATA_PROCESSED, EXPERIMENT_PARQUET


def _connect() -> duckdb.DuckDBPyConnection:
    tmp = DATA_PROCESSED / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    return con


def query_experiment(sql: str) -> pd.DataFrame:
    """Run one SELECT against the ``experiment`` view.

    Args:
        sql: SQL that reads from ``experiment``.

    Returns:
        Query result.
    """
    if not EXPERIMENT_PARQUET.exists():
        raise FileNotFoundError("Experiment parquet has not been written.")
    con = _connect()
    con.execute(
        f"CREATE VIEW experiment AS SELECT * FROM read_parquet('{EXPERIMENT_PARQUET.as_posix()}')"
    )
    frame = con.execute(sql).fetchdf()
    con.close()
    return frame


def read_sql_file(path: Path) -> str:
    """Read a SQL file, dropping full-line comments and a trailing semicolon."""
    lines = [line for line in path.read_text().splitlines() if not line.strip().startswith("--")]
    return "\n".join(lines).strip().rstrip(";")
