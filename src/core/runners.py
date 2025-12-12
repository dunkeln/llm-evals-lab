from datetime import datetime, timezone

from src.core.defs import GenerationResult, ModelRunner, TaskType, load_env
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.config import PROJECT_ROOT
import os

from typing import Any

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


class OpenAIRunner(ModelRunner):
    def __init__(self, model="gpt-4.1", provider="openai", task=TaskType.Completion):
        self.model = model
        self.provider = provider
        self.client = ChatOpenAI(model=self.model, api_key=load_env("OPENAI_API_KEY"))
        self.task = task

    def generate(self, query: str, process_fn=None) -> GenerationResult:
        resp = self.client.invoke(query)
        text = process_fn(resp) if process_fn is not None else resp.content

        if not isinstance(text, str):
            raise ValueError("process_fn should parse content to `str` type only")

        now = datetime.now(timezone.utc).isoformat()
        return GenerationResult(
            model=self.model,
            provider=self.provider,
            task=self.task.value if hasattr(self.task, "value") else str(self.task),
            response_text=text,
            response_metadata=getattr(resp, "response_metadata", {}),
            usage_metadata=getattr(resp, "usage_metadata", {}),
            run_timestamp=now
        )


class ClaudeRunner(ModelRunner):
    def __init__(self, model="claude-sonnet-4-5-20250929", provider="claude", task=TaskType.Completion):
        self.model = model
        self.provider = provider
        self.client = ChatAnthropic(model_name=self.model, api_key=load_env("CLAUDE_API_KEY"))          # type: ignore
        self.task = task

    def generate(self, query: str, process_fn=None) -> GenerationResult:
        resp = self.client.invoke(query)
        text = process_fn(resp) if process_fn is not None else resp.content

        if not isinstance(text, str):
            raise ValueError("process_fn should parse content to `str` type only")

        now = datetime.now(timezone.utc).isoformat()
        return GenerationResult(
            model=self.model,
            provider=self.provider,
            task=self.task.value if hasattr(self.task, "value") else str(self.task),
            response_text=text,
            response_metadata=getattr(resp, "response_metadata", {}),
            usage_metadata=getattr(resp, "usage_metadata", {}),
            run_timestamp=now
        )

class GeminiRunner(ModelRunner):
    def __init__(self, model="gemini-2.5-flash-lite", provider="google", task=TaskType.Completion):
        self.model = model
        self.provider = provider
        self.client = ChatGoogleGenerativeAI(model=self.model, api_key=load_env("GEMINI_API_KEY"))
        self.task = task

    def generate(self, query: str, process_fn=None) -> GenerationResult:
        resp = self.client.invoke(query)
        text = process_fn(resp) if process_fn is not None else resp.content

        if not isinstance(text, str):
            raise ValueError("process_fn should parse content to `str` type only")

        now = datetime.now(timezone.utc).isoformat()
        return GenerationResult(
            model=self.model,
            provider=self.provider,
            task=self.task.value if hasattr(self.task, "value") else str(self.task),
            response_text=text,
            response_metadata=getattr(resp, "response_metadata", {}),
            usage_metadata=getattr(resp, "usage_metadata", {}),
            run_timestamp=now
        )

if __name__ == "__main__":
        gpt = OpenAIRunner()
        claude = ClaudeRunner()
        gemini = GeminiRunner()
        print(gpt.generate("I am batman"))
        print(claude.generate("I am batman"))
        print(gemini.generate("I am batman"))
