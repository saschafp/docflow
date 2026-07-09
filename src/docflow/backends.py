from dataclasses import dataclass
from typing import Protocol

from litellm import completion


class LLMBackend(Protocol):
    """
    Interface for LLM backends.
    """

    name: str

    def complete(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str: ...


class DummyBackend:
    """
    Dummy backend for tests and examples.
    """

    name = "dummy"

    def complete(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        return '{"label": "dummy", "rationale": "This is a dummy response."}'


@dataclass(frozen=True)
class LiteLLMBackend:
    """
    LLM backend using LiteLLM.
    """

    model: str
    url: str | None = None
    token: str | None = None
    temperature: float = 0.0

    @property
    def name(self) -> str:
        return f"litellm:{self.model}"

    def complete(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        messages = []

        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt is not None:
            messages.append({"role": "user", "content": user_prompt})

        response = completion(
            model=self.model,
            messages=messages,
            api_base=self.url,
            api_key=self.token,
            temperature=self.temperature,
        )

        content = response.choices[0].message.content

        if content is None:
            raise RuntimeError("Received empty response from LLM.")

        return content
