from dataclasses import dataclass
from typing import Any, cast


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
        try:
            from litellm import completion
        except ImportError as error:
            raise ImportError(
                "LiteLLMBackend requires litellm. "
                "Install it with `pip install docflow[litellm]`."
            ) from error

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

        return _completion_text(response)


def _completion_text(response: object) -> str:
    typed_response = cast(Any, response)

    content = typed_response.choices[0].message.content

    if content is None:
        raise RuntimeError("Received empty response from LLM.")

    return str(content)
