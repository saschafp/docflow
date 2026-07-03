from typing import Protocol


class LLMBackend(Protocol):
    name: str

    def complete(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str: ...


class DummyBackend:
    name = "dummy"

    def complete(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        return '{"label": "dummy", "rationale": "This is a dummy response."}'
