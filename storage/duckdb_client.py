from contextlib import contextmanager
import duckdb
from pathlib import Path
from src.config import PROJECT_ROOT

# PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "warehouse" / "llm_evals.duckdb"

def get_connection(read_only: bool=False) -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH), read_only=read_only)

@contextmanager
def duckdb_session(read_only: bool = False):
    conn = get_connection(read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()
