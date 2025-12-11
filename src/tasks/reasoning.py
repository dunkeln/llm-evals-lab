from typing import Any
from .base import EvalTask

class ReasoningTask(EvalTask):
    name = "reasoning"
    
    def load_dataset(self, split: str) -> list[dict]:
        ...
    
    def build_prompt(self, example: dict) -> str:
        ...
