select * from {{ ref('models_reasoning_scores') }}
  where token_f1 < 0 or token_f1 > 1
    or cosine_similarity < 0 or cosine_similarity > 1
