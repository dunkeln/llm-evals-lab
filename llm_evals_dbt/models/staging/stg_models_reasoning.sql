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
  (usage_metadata->>'latency')::int as latency,
  (usage_metadata->>'temperature')::float as temperature,
  (usage_metadata->>'max_tokens')::int as max_tokens,
  (usage_metadata->>'top_p')::int as top_p,
  (metrics->>'cos_sim')::float as cosine_similarity
from {{ source('raw', 'models_reasoning') }}
