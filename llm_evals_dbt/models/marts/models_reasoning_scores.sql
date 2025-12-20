-- INFO: normalize all data

{{
  config(
    materialized='incremental',
    unique_key=['id'],
    on_schema_change='append_new_columns'
  )
}}
with results as (
  select *
  from {{ ref('stg_models_reasoning') }}
  {% if is_incremental() %}
    where id not in (select distinct id from {{ this }} )
    and run_timestamp >= (select max(run_timestamp) from {{ this }} )
  {% endif %}
),
dataset as (
  select *
  from {{ ref('stg_models_reasoning_data') }}
  {%if is_incremental() %}
    where id not in (select distinct id from {{ this }} )
  {% endif %}
),
metadata as (
  select *
  from {{ ref('stg_models_reasoning_metadata') }}
)
select 
    d.id, d.question, d.answer, d.explanation, r.response_text, d.difficulty, d.metadata,
    r.run_timestamp, r.model, r.provider, r.input_tokens, r.output_tokens, r.total_tokens, r.latency,
    r.temperature, r.max_tokens, r.top_p, r.toxicity, r.verbosity, r.hallucination, r.correctness, r.cosine_similarity, r.token_f1,
    m.model as s_model, m.temperature as s_temp, m.top_p as s_top_p, m.max_tokens as s_max_tokens
  from dataset d
  join results r
  on r.id = d.id
  join metadata m
  on m.id = d.metadata
  order by r.run_timestamp
