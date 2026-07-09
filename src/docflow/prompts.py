from dataclasses import dataclass


@dataclass
class PromptTemplate:
    text: str

    def partial(self, **kwargs: object) -> "PromptTemplate":
        text = self.text

        for key, value in kwargs.items():
            text = text.replace(f"{{{key}}}", str(value))

        return PromptTemplate(text=text)

    def render(self, **kwargs: object) -> str:
        text = self.text

        for key, value in kwargs.items():
            text = text.replace(f"{{{key}}}", str(value))

        return text