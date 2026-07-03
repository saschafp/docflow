from .backends import LLMBackend
from .schemas import Document, Prediction

SYSTEM_PROMPT = """
You are a strict document classifier.
Return exactly one label from the allowed labels.
Return only valid JSON with the keys "label" and "rationale".
"""


def build_user_prompt(document: Document, labels: list[str]) -> str:
    return f"""
Classify the following document.

Allowed labels
{labels}

Document:
{document.text}
"""


def classify_one(
    document: Document,
    backend: LLMBackend,
    labels: list[str],
) -> Prediction:
    user_prompt = build_user_prompt(document, labels)
    raw = backend.complete(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    return Prediction(
        document_id=document.id,
        label=labels[0],  # TODO
        rationale=raw,
    )


def classify(
    documents: list[Document],
    backend: LLMBackend,
    labels: list[str],
) -> list[Prediction]:
    return [classify_one(document, backend, labels) for document in documents]
