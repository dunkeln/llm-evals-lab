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
select d.*, r.* 
  from dataset d
  join results r
  on r.id = d.id
  join metadata m
  on m.id = d.metadata
  order by r.run_timestamp
