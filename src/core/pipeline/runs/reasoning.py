import json
from src.core.runners import OpenAIRunner, ClaudeRunner, GeminiRunner, SynthesizerModel
from src.core.defs import GenerationResult, JSONLResponse
from src.core.pipeline.ingest import ingest_jsonl_to_raw
from src.core.pipeline.writers import write_result_to_jsonl


if __name__ == '__main__':
    synthesizer = SynthesizerModel(model="gpt-4o")
    runners = [
        OpenAIRunner(),
        ClaudeRunner(),
        GeminiRunner(),
    ]

    prompt = "Generate basic Statistics questions to solve."
    dataset = synthesizer.generate(prompt, batch_size=2)
    samples = dataset.samples

    records: list[GenerationResult] = []
    for sample in samples:
        rows: list[GenerationResult] = []
        query = sample.question
        id = sample.id
        for runner in runners:
            rec = runner.generate(id, query)
            print(rec)
            rows.append(rec)

        records.extend(rows)

    dataset_path, jsonl_path = write_result_to_jsonl(dataset, records)
    ingest_jsonl_to_raw(dataset_path, jsonl_path, table_name="models_reasoning")

    print("done")
