from typing import Protocol


class LLMBackend(Protocol):
    """Interface for LLM backends."""

    @property
    def name(self) -> str:
        """Backend name."""
        ...

    def complete(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        """Generate text from optional system and user prompts."""
        ...
