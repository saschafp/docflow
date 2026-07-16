from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaBackend:
    """Ollama backend using httpx."""

    model: str
    url: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout: float = 120.0

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def complete(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        try:
            import httpx
        except ImportError as error:
            raise ImportError(
                "OllamaBackend requires httpx. "
                "Install it with `pip install docflow[ollama]`."
            ) from error

        messages: list[dict[str, str]] = []

        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        if user_prompt is not None:
            messages.append({"role": "user", "content": user_prompt})

        if not messages:
            raise ValueError("At least one prompt must be provided.")

        response = httpx.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                },
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content")

        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response: {data}")

        return content
