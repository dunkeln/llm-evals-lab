from src.core.runners import DeepseekRunner, OpenAIRunner, ClaudeRunner, GeminiRunner, SynthesizerModel
from src.core.defs import GenerationResult, JSONLResponse
from src.core.pipeline.ingest import ingest_jsonl_to_raw
from src.core.pipeline.writers import write_result_to_jsonl
from src.config import log

import asyncio
import time

logger = log(__name__)

async def main() -> None:
    batch_size = 2
    synthesizer = SynthesizerModel(model="gpt-4.1")
    max_in_flight = 8

    runners = [
        OpenAIRunner(model="gpt-4o-mini"),
        ClaudeRunner(model="claude-haiku-4-5-20251001", temperature=0.1),
        GeminiRunner(model="gemini-2.0-flash"),
        DeepseekRunner(model="deepseek-chat")
    ]

    logger.info(f"{len(runners)} models loaded...")

    prompt = "Generate questions on bayesian reasoning/ base-rate problems."
    dataset_start = time.perf_counter()
    dataset = synthesizer.generate(prompt, total_samples=batch_size, per_batch=5)
    dataset_elapsed = time.perf_counter() - dataset_start
    samples = dataset.samples

    logger.info(f"{batch_size} samples generated....")
    logger.info(f"dataset generation time: {dataset_elapsed:.2f}s")


    records: list[GenerationResult] = []
    logger.info(f"evaluating models on samples...")
    total_loop_elapsed = 0.0
    eval_start = time.perf_counter()
    semaphore = asyncio.Semaphore(max_in_flight)

    async def run_model(runner, sample_id, query, ref_answer):
        async with semaphore:
            return await runner.generate(sample_id, query, ref_answer)

    async def run_sample(sample, idx):
        loop_start = time.perf_counter()
        query = sample.question
        ref_answer = sample.answer
        sample_id = sample.id
        rows = await asyncio.gather(
            *(run_model(runner, sample_id, query, ref_answer) for runner in runners)
        )
        loop_elapsed = time.perf_counter() - loop_start
        logger.info(
            f"generation complete on sample {idx + 1}/{len(samples)} "
            f"({loop_elapsed:.2f}s)"
        )
        return rows, loop_elapsed

    sample_tasks = [
        asyncio.create_task(run_sample(sample, idx))
        for idx, sample in enumerate(samples)
    ]
    for task in asyncio.as_completed(sample_tasks):
        rows, loop_elapsed = await task
        total_loop_elapsed += loop_elapsed
        records.extend(rows)

    eval_elapsed = time.perf_counter() - eval_start
    logger.info(f"Models ran for {len(samples) * len(runners)}...")
    logger.info(f"total per-sample loop time: {total_loop_elapsed:.2f}s")
    logger.info(f"total wall time: {eval_elapsed:.2f}s")
    metadata_path, dataset_path, jsonl_path = write_result_to_jsonl(dataset, records)
    logger.info(f"Ingesting samples to jsonl")
    logger.info(f"metadata   path: {metadata_path}")
    logger.info(f"samples    path: {dataset_path}")
    logger.info(f"generation path: {dataset_path}")
    ingest_jsonl_to_raw(metadata_path, dataset_path, jsonl_path, table_name="models_reasoning")
    logger.info("Finished.")


if __name__ == '__main__':
    asyncio.run(main())
