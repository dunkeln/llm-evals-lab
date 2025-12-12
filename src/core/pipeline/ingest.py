from pathlib import Path
from storage.duckdb_client import duckdb_session
import polars as pl

def ingest_jsonl_to_raw(jsonl_path: Path, table_name: str | None) -> None:
    df = pl.read_ndjson(jsonl_path)
    if table_name is None:
        # TODO: table name can be predefined or be the same name as jsonl
        ...

    with duckdb_session() as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS raw;")
        conn.execute(f"DROP TABLE IF EXISTS raw.{table_name}")
        conn.register("tmp_df", df)
        conn.execute(f"CREATE TABLE raw.{table_name} AS SELECT * FROM tmp_df;")

