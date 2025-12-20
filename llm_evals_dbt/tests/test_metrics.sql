select * from {{ ref('models_reasoning_scores') }}
  where token_f1 < 0 or token_f1 > 1
    or cosine_similarity < 0 or cosine_similarity > 1
    or verbosity < 1 or verbosity > 10
    or toxicity < 0 or toxicity > 10
    or hallucination < 0 or hallucination > 10
    or correctness < 0 or correctness > 1
    or total_tokens <> input_tokens + output_tokens
    or latency < 0 or latency is null
