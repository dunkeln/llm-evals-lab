

### Project README


```text
         ┌─────────────────────────────┐
         │          src/               │
         │  (LLM eval app logic)       │
         │                             │
         │  run_models.py              │
         │    └→ calls APIs            │
         │       writes JSONL          │
         │                             │
         │  ingest_results.py          │
         │    └→ reads JSONL           │
         │       loads into DuckDB     │
         └────────────┬────────────────┘
                      │
         data/raw/    │    warehouse/llm_eval.duckdb
    (landing zone)    │    (DuckDB file with raw.summarization, ...)
                      │
                      ▼
              dbt/llm_eval_dbt/
                (Transforms)
              models/raw/*.sql       – trivial selects over raw.*
              models/marts/*.sql     – metrics, aggregates
```
