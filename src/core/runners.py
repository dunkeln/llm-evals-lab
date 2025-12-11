from src.core.defs import ModelRunner, TaskType, load_env
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

    def generate(self, query: str) -> Any:
        return self.client.invoke(query)

class ClaudeRunner(ModelRunner):
    def __init__(self, model="claude-sonnet-4-5-20250929", provider="claude", task=TaskType.Completion):
        self.model = model
        self.provider = provider
        self.client = ChatAnthropic(model_name=self.model, api_key=load_env("CLAUDE_API_KEY"))          # type: ignore
        self.task = task

    def generate(self, query: str) -> Any:
        return self.client.invoke(query)

class GeminiRunner(ModelRunner):
    def __init__(self, model="gemini-2.5-flash-lite", provider="google", task=TaskType.Completion):
        self.model = model
        self.provider = provider
        self.client = ChatGoogleGenerativeAI(model=self.model, api_key=load_env("GEMINI_API_KEY"))
        self.task = task

    def generate(self, query: str) -> Any:
        return self.client.invoke(query)

gpt = OpenAIRunner()
claude = ClaudeRunner()
gemini = GeminiRunner()
print(gpt.generate("I am batman"))
print(claude.generate("I am batman"))
print(gemini.generate("I am batman"))
