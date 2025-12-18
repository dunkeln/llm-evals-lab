import json
from src.core.runners import OpenAIRunner, ClaudeRunner, GeminiRunner, SynthesizerModel
from src.core.defs import GenerationResult, JSONLResponse
from src.core.pipeline.ingest import ingest_jsonl_to_raw
from src.core.pipeline.writers import write_result_to_jsonl


if __name__ == '__main__':
    synthesizer = SynthesizerModel(model="gpt-4o")
    runners = [
        OpenAIRunner(),
        ClaudeRunner(temperature=0.1),
        GeminiRunner(),
    ]

    prompt = "Generate basic Statistics questions to solve."
    dataset = synthesizer.generate(prompt, batch_size=2)
    samples = dataset.samples

    records: list[GenerationResult] = []
    for sample in samples:
        rows: list[GenerationResult] = []
        query = sample.question
        ref_answer = sample.answer
        id = sample.id
        for runner in runners:
            rec = runner.generate(id, query, ref_answer)
            rows.append(rec)

        records.extend(rows)

    print("dataset")
    print(dataset)

    metadata_path, dataset_path, jsonl_path = write_result_to_jsonl(dataset, records)
    ingest_jsonl_to_raw(metadata_path, dataset_path, jsonl_path, table_name="models_reasoning")

    print("done")
