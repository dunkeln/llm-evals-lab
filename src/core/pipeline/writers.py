from dataclasses import asdict
from typing import Tuple
from src.core.defs import GenerationResult, JSONLResponse
from src.config import PROJECT_ROOT
from pathlib import Path
import json
from datetime import datetime, timezone

DUMP_DIR = PROJECT_ROOT / "data" / "raw"

def write_result_to_jsonl(
    dataset: JSONLResponse,
    results: list[GenerationResult]
) -> Tuple[Path, Path]:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).isoformat()

    samples_path = DUMP_DIR / f"{results[0].task}-data-{run_ts}.jsonl"
    out_path = DUMP_DIR / f"{results[0].task}-{run_ts}.jsonl"

    with samples_path.open("w") as f:
        for sample in dataset.samples:
            record = sample.model_dump()
            f.write(json.dumps(record) + "\n")

    with out_path.open("w") as f:
        for res in results:
            record = {
                **asdict(res)
            }
            f.write(json.dumps(record) + "\n")

    return samples_path, out_path
