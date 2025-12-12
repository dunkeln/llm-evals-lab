from dataclasses import asdict
from src.core.defs import GenerationResult
from src.config import PROJECT_ROOT
from pathlib import Path
import json
from datetime import datetime

DUMP_DIR = PROJECT_ROOT / "data" / "raw"

def write_result_to_jsonl(
    results: list[GenerationResult]
) -> Path:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DUMP_DIR / f"{results[0].task}-{datetime.now()}.jsonl"

    with out_path.open("w") as f:
        for res in results:
            record = {
                **asdict(res)
            }
            f.write(json.dumps(record) + "\n")

    return out_path
