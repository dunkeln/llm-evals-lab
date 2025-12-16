from datetime import datetime, timezone
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
    def __init__(self, model: str, output_model: Type[T]=JSONLResponse) -> None:
        self.model = model
        self.client = ChatOpenAI(model=self.model, api_key=load_env("OPENAI_API_KEY"))
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

        result = self.structured_client.invoke(messages)
        for sample in result.samples:
            sample.id = str(uuid.uuid4())
        return result

class OpenAIRunner(ModelRunner):
    def __init__(self, model="gpt-4.1", provider="openai", task=TaskType.Completion):
        self.model = model
        self.provider = provider
        self.client = ChatOpenAI(model=self.model, api_key=load_env("OPENAI_API_KEY"))
        self.task = task

    def generate(self, id: str, query: str, answer: str, process_fn=None) -> GenerationResult:
        resp = self.client.invoke(query)
        text = process_fn(resp) if process_fn is not None else resp.content

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


class ClaudeRunner(ModelRunner):
    def __init__(self, model="claude-sonnet-4-5-20250929", provider="claude", task=TaskType.Completion):
        self.model = model
        self.provider = provider
        self.client = ChatAnthropic(model_name=self.model, api_key=load_env("CLAUDE_API_KEY"))          # type: ignore
        self.task = task

    def generate(self, id: str, query: str, answer: str, process_fn=None) -> GenerationResult:
        resp = self.client.invoke(query)
        text = process_fn(resp) if process_fn is not None else resp.content

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
    def __init__(self, model="gemini-2.5-flash-lite", provider="google", task=TaskType.Completion):
        self.model = model
        self.provider = provider
        self.client = ChatGoogleGenerativeAI(model=self.model, api_key=load_env("GEMINI_API_KEY"))
        self.task = task

    def generate(self, id: str, query: str, answer: str, process_fn=None) -> GenerationResult:
        resp = self.client.invoke(query)
        text = process_fn(resp) if process_fn is not None else resp.content

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
