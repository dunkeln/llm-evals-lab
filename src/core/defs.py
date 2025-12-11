from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Protocol
import os

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

class TaskType(Enum):
    Summarization = "Summarization"
    Reasoning = "Reasoning"
    Math = "Math"
    Completion = "Completion"
 
@dataclass
class ModelRunner(Protocol):
    model: str
    provider: str
    task: TaskType
    client: ChatGoogleGenerativeAI | ChatOpenAI | ChatAnthropic


    def generate(self, query: str) -> str:
        ...

@dataclass
class GenerationResult(Protocol):
    model: str
    provider: str
    task: TaskType
    client: ChatAnthropic | ChatOpenAI | ChatGoogleGenerativeAI
    response_text: str
    response_metadata: Dict[str, Any]
    usage_metadata: Dict[str, Any]
    run_timestamp: str                                                      # INFO: ISO string

def load_env(name: str) -> SecretStr:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Requied env {name} is not set.")

    return SecretStr(value)
