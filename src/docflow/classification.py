import json

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

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Prediction(
            document_id=document.id,
            label="invalid_response",
            rationale=raw,
        )

    label = data.get("label", "No label provided.")
    rationale = data.get("rationale", "No rationale provided.")

    if label not in labels:
        invalid_label = label
        label = "invalid_label"
        rationale = f"Invalid label returned: {invalid_label!r}. Raw response: {raw!r}"

    return Prediction(
        document_id=document.id,
        label=label,
        rationale=rationale,
    )


def classify(
    documents: list[Document],
    backend: LLMBackend,
    labels: list[str],
) -> list[Prediction]:
    return [classify_one(document, backend, labels) for document in documents]
