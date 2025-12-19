from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Protocol
import os

from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

class TaskType(Enum):
    Summarization = "Summarization"
    Reasoning = "Reasoning"
    Math = "Math"
    Completion = "Completion"
 
class JSONResponse(BaseModel):
    id: str = Field(..., description="rolling identifier for the question only as an integer value.")
    question: str
    answer: str
    explanation: str
    difficulty: str
    metadata: str

class JSONLResponse(BaseModel):
    id: str
    batch: int = Field(..., description="number of samples")
    model: str
    temperature: float
    top_p: float
    max_tokens: float
    samples: list[JSONResponse]

@dataclass
class ReasoningResult:
    example_id: str
    reference: str
    answer: str
    metrics: Dict[str, float]

@dataclass
class GenerationResult:
    id: str
    model: str
    provider: str
    task: str
    response_text: str
    response_metadata: Dict[str, Any]
    usage_metadata: Dict[str, Any]
    run_timestamp: str                                                      # INFO: ISO string
    reasoning_result: ReasoningResult | None = None
    metrics: Optional[Dict[str, float]] = None


def load_env(name: str) -> SecretStr:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Requied env {name} is not set.")

    return SecretStr(value)

ParseFn = Callable[[Any], str]

@dataclass
class ModelRunner(Protocol):
    model: str
    provider: str
    task: TaskType
    client: ChatGoogleGenerativeAI | ChatOpenAI | ChatAnthropic | ChatDeepSeek


    def generate(self, id: str, query: str, answer: str, process_fn: ParseFn | None=None) -> GenerationResult:
        ...
