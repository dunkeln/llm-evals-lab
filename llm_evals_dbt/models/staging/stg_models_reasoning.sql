select
  id,
  run_timestamp,
  task,
  model,
  provider,
  response_text,
  (usage_metadata->>'input_tokens')::int as input_tokens,
  (usage_metadata->>'output_tokens')::int as output_tokens,
  (usage_metadata->>'input_tokens')::int + (usage_metadata->>'output_tokens')::int as total_tokens,
  (metrics->>'cos_sim')::float as cosine_similarity
from {{ source('raw', 'models_reasoning') }}
