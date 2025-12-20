from src.core.runners import DeepseekRunner, OpenAIRunner, ClaudeRunner, GeminiRunner, SynthesizerModel
from src.core.defs import GenerationResult, JSONLResponse
from src.core.pipeline.ingest import ingest_jsonl_to_raw
from src.core.pipeline.writers import write_result_to_jsonl
from src.config import log

logger = log(__name__)

if __name__ == '__main__':
    batch_size = 5
    synthesizer = SynthesizerModel(model="gpt-4o")
    runners = [
        OpenAIRunner(),
        ClaudeRunner(temperature=0.1),
        GeminiRunner(),
        DeepseekRunner()
    ]

    logger.info(f"{len(runners)} models loaded...")

    prompt = "Generate questions on bayesian reasoning/ base-rate problems."
    dataset = synthesizer.generate(prompt, total_samples=batch_size, per_batch=5)
    samples = dataset.samples

    logger.info(f"{batch_size} samples generated....")

    records: list[GenerationResult] = []
    logger.info(f"evaluating models on samples...") 
    for idx, sample in enumerate(samples):
        rows: list[GenerationResult] = []
        query = sample.question
        ref_answer = sample.answer
        id = sample.id
        for runner in runners:
            rec = runner.generate(id, query, ref_answer)
            rows.append(rec)

        logger.info(f"generation complete on sample {idx + 1}/{len(samples)}")
        records.extend(rows)

    logger.info(f"Models ran for {len(samples) * len(runners)}...")
    metadata_path, dataset_path, jsonl_path = write_result_to_jsonl(dataset, records)
    logger.info(f"Ingesting samples to jsonl")
    logger.info(f"metadata   path: {metadata_path}")
    logger.info(f"samples    path: {dataset_path}")
    logger.info(f"generation path: {dataset_path}")
    ingest_jsonl_to_raw(metadata_path, dataset_path, jsonl_path, table_name="models_reasoning")
    logger.info("Finished.")
