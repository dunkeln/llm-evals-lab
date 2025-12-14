from pathlib import Path
from storage.duckdb_client import duckdb_session
import polars as pl

def ingest_jsonl_to_raw(samples_path: Path, jsonl_path: Path, table_name: str | None) -> None:
    df = pl.read_ndjson(jsonl_path)
    data_df = pl.read_ndjson(samples_path)

    with duckdb_session() as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS raw;")
        conn.register("tmp_df", df)
        conn.execute(f"CREATE TABLE IF NOT EXISTS raw.{table_name} AS SELECT * FROM tmp_df LIMIT 0;")
        conn.execute(f"INSERT INTO raw.{table_name} SELECT * FROM tmp_df")
        conn.register("tmp_data_df", data_df)
        conn.execute(f"CREATE TABLE IF NOT EXISTS raw.{table_name}_data AS SELECT * FROM tmp_data_df LIMIT 0;")
        conn.execute(f"INSERT INTO raw.{table_name}_data SELECT * FROM tmp_data_df")
