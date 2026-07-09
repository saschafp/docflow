import json

from .schemas import Classification, Response


def classification_from_response(
    response: Response,
    labels: list[str],
) -> Classification:
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        return Classification(
            document_id=response.document_id,
            label="invalid_json",
            rationale=f"Response was not valid JSON: {response.text!r}",
        )

    if not isinstance(data, dict):
        return Classification(
            document_id=response.document_id,
            label="invalid_json",
            rationale=f"Response was not a JSON object: {response.text!r}",
        )

    label = data.get("label", "missing_label")
    rationale = data.get("rationale", "missing_rationale")

    if label not in labels:
        invalid_label = label
        return Classification(
            document_id=response.document_id,
            label="invalid_label",
            rationale=f"Invalid label returned: {invalid_label!r}. Raw response: {response.text!r}",
        )

    return Classification(
        document_id=response.document_id,
        label=label,
        rationale=rationale,
    )


def classifications_from_responses(
    responses: list[Response],
    labels: list[str],
) -> list[Classification]:
    return [
        classification_from_response(response=response, labels=labels)
        for response in responses
    ]
