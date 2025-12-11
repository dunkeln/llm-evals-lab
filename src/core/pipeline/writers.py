from typing import Iterable
from src.core.defs import GenerationResult
from src.config import PROJECT_ROOT
from pathlib import Path
import json

DUMP_DIR = PROJECT_ROOT / "data" / "raw"

def write_result_to_jsonl(
    results: Iterable[GenerationResult]
) -> Path:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    outpath = DUMP_DIR / f"{results[0].task}.jsonl"

    with outpath.open("w") as f:
        for res in results:
            record = {
            }
    ...
