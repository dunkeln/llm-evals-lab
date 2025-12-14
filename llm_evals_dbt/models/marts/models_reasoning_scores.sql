with results as (
  select *
  from {{ ref('stg_models_reasoning') }}
),
dataset as (
  select *
  from {{ ref('stg_models_reasoning_data') }}
)

select d.*, r.* from dataset d join results r on r.id = d.id order by r.run_timestamp
