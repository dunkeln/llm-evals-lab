from datetime import datetime, timezone
import time
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.core.defs import GenerationResult, JSONLResponse, ModelRunner, TaskType, load_env
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.config import PROJECT_ROOT
import os

from typing import Any, Generic, Type, TypeVar, cast

from src.core.metrics import get_metrics

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


T = TypeVar("T", bound=BaseModel)

class SynthesizerModel(Generic[T]):
    # TODO: for now chatgpt is the only model, need to be provider agnostic
    def __init__(self, model: str, output_model: Type[T]=JSONLResponse, temperature=0.0, top_p=1.0, max_tokens=512) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = ChatOpenAI(
            model=self.model,
            api_key=load_env("OPENAI_API_KEY"),
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_tokens
        )
        self.output_model = output_model
        self.structured_client = self.client.with_structured_output(self.output_model)

    def generate(self, query: str, batch_size: int=10):
        system_prompt = f"""
        You are a data synthesis assistant. Follow the instructions in the user
        message and generate clean, well-structured text suitable as part of an
        evaluation dataset. Do not answer anything except the output for the
        requested content in jsonl. Generate {batch_size} such content.

        REMEMBER: output *ONLY* jsonl object of size {batch_size}, no extra text.
        """
        system_prompt = system_prompt.strip()
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=query))

        result: JSONLResponse = cast(JSONLResponse, self.structured_client.invoke(messages))
        result.id = str(uuid.uuid4())
        for sample in result.samples:
            sample.id = str(uuid.uuid4())
            sample.metadata = result.id
        result.model = self.model
        result.temperature = self.temperature
        result.top_p = self.top_p
        result.max_tokens = self.max_tokens
        return result

class OpenAIRunner(ModelRunner):
    def __init__(self, model="gpt-4.1", provider="openai", task=TaskType.Completion, temperature=0.0, top_p=1.0, max_tokens=512):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = ChatOpenAI(
            model=self.model,
            api_key=load_env("OPENAI_API_KEY"),
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_tokens
        )
        self.task = task

    def generate(self, id: str, query: str, answer: str, process_fn=None) -> GenerationResult:
        start = time.time()
        resp = self.client.invoke(query)
        end = time.time()
        text = process_fn(resp) if process_fn is not None else resp.content
        usage = getattr(resp, 'usage_metadata')
        usage['latency'] = end - start
        usage['temperature'] = self.temperature
        usage['max_tokens'] = self.max_tokens
        usage['top_p'] = self.top_p

        if not isinstance(text, str):
            raise ValueError("process_fn should parse content to `str` type only")

        now = datetime.now(timezone.utc).isoformat()
        return GenerationResult(
            id=id,
            model=self.model,
            provider=self.provider,
            task=self.task.value if hasattr(self.task, "value") else str(self.task),
            response_text=text,
            response_metadata=getattr(resp, "response_metadata", {}),
            usage_metadata=getattr(resp, "usage_metadata", {}),
            run_timestamp=now,
            metrics=get_metrics(text, answer),
        )


class ClaudeRunner(ModelRunner):
    def __init__(self, model="claude-sonnet-4-5-20250929", provider="claude", task=TaskType.Completion, temperature=None, top_p=None, max_tokens=512):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.top_p = top_p
        if (self.temperature is None and self.top_p is None) or (self.temperature is not None and self.top_p is not None):
            raise ValueError("Claude requires atleast one of temperature or top_p to be set")
        self.max_tokens = max_tokens
        self.client = ChatAnthropic(
            model_name=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens_to_sample=self.max_tokens,
            api_key=load_env("CLAUDE_API_KEY")
        ) # type: ignore
        self.task = task

    def generate(self, id: str, query: str, answer: str, process_fn=None) -> GenerationResult:
        start = time.time()
        resp = self.client.invoke(query)
        end = time.time()
        text = process_fn(resp) if process_fn is not None else resp.content
        usage = getattr(resp, 'usage_metadata')
        usage['latency'] = end - start
        usage['temperature'] = self.temperature
        usage['max_tokens'] = self.max_tokens
        usage['top_p'] = self.top_p

        if not isinstance(text, str):
            raise ValueError("process_fn should parse content to `str` type only")

        now = datetime.now(timezone.utc).isoformat()
        return GenerationResult(
            id=id,
            model=self.model,
            provider=self.provider,
            task=self.task.value if hasattr(self.task, "value") else str(self.task),
            response_text=text,
            response_metadata=getattr(resp, "response_metadata", {}),
            usage_metadata=getattr(resp, "usage_metadata", {}),
            run_timestamp=now,
            metrics=get_metrics(text, answer)
        )

class GeminiRunner(ModelRunner):
    def __init__(self, model="gemini-2.5-flash-lite", provider="google", task=TaskType.Completion, temperature=0.0, top_p=1.0, max_tokens=512):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            api_key=load_env("GEMINI_API_KEY"),
        )
        self.task = task

    def generate(self, id: str, query: str, answer: str, process_fn=None) -> GenerationResult:
        start = time.time()
        resp = self.client.invoke(query)
        end = time.time()
        text = process_fn(resp) if process_fn is not None else resp.content
        usage = getattr(resp, 'usage_metadata')
        usage['latency'] = end - start
        usage['temperature'] = self.temperature
        usage['max_tokens'] = self.max_tokens
        usage['top_p'] = self.top_p

        if not isinstance(text, str):
            raise ValueError("process_fn should parse content to `str` type only")

        now = datetime.now(timezone.utc).isoformat()
        return GenerationResult(
            id=id,
            model=self.model,
            provider=self.provider,
            task=self.task.value if hasattr(self.task, "value") else str(self.task),
            response_text=text,
            response_metadata=getattr(resp, "response_metadata", {}),
            usage_metadata=getattr(resp, "usage_metadata", {}),
            run_timestamp=now,
            metrics=get_metrics(text, answer)
        )

if __name__ == "__main__":
    # gpt = OpenAIRunner()
    # claude = ClaudeRunner()
    # gemini = GeminiRunner()
    # print(gpt.generate("I am batman"))
    # print(claude.generate("I am batman"))
    # print(gemini.generate("I am batman"))
    synthesizer = SynthesizerModel(model="gpt-4o", output_model=JSONLResponse)
    prompt = """
    Generate reasoning questions about basic statistics.
    """

    result = synthesizer.generate(prompt, batch_size=2)
    print(result)
