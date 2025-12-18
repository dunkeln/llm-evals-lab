select *
from {{ source('raw', 'models_reasoning_metadata') }}
