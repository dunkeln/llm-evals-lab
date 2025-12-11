from typing import Protocol


class EvalTask(Protocol):
    name: str

    def load_dataset(self, split: str) -> list[dict]:
        ...

    def build_prompt(self, example: dict) -> str:
        ...

    def extract_answer(self, raw_output: str) -> str:
        ...
