select
  run_id,
  run_timestamp,
  task,
  model as model_name,
  provider,
  example_id,
  reference,
  parsed_answer,
  metrics,
  usage,
  response_metadata
from {{ source('raw', 'reasoning_results') }}
