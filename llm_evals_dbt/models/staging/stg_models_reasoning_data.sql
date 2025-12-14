select
  id,
  question,
  answer,
  explanation,
  difficulty
from {{ source('raw', 'models_reasoning_data') }}
