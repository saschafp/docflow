from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """
    Simple prompt template with partial and final rendering.
    """

    text: str

    def partial(self, **kwargs: object) -> "PromptTemplate":
        """
        Return a new PromptTemplate with selected placeholders replaced.
        """
        text = self.text

        for key, value in kwargs.items():
            text = text.replace(f"{{{key}}}", str(value))

        return PromptTemplate(text=text)

    def render(self, **kwargs: object) -> str:
        """
        Render the template by replacing placeholders with values.
        """
        text = self.text

        for key, value in kwargs.items():
            text = text.replace(f"{{{key}}}", str(value))

        return text
