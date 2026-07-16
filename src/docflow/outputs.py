import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schemas import Classification, Document, Response


def documents_to_dicts(documents: list[Document]) -> list[dict[str, Any]]:
    """Convert documents to dictionaries."""
    return [_as_csv_dict(document) for document in documents]


def responses_to_dicts(responses: list[Response]) -> list[dict[str, Any]]:
    """Convert responses to dictionaries."""
    return [_as_csv_dict(response) for response in responses]


def classifications_to_dicts(
    classifications: list[Classification],
) -> list[dict[str, Any]]:
    """Convert classifications to dictionaries."""
    return [_as_csv_dict(classification) for classification in classifications]


def save_documents_csv(
    documents: list[Document],
    path: str | Path,
) -> None:
    """Save documents to a CSV file."""
    rows = documents_to_dicts(documents)
    _save_csv(
        rows=rows,
        path=path,
        fieldnames=["id", "path", "text"],
    )


def save_responses_csv(
    responses: list[Response],
    path: str | Path,
) -> None:
    """Save responses to a CSV file."""
    rows = responses_to_dicts(responses)
    _save_csv(
        rows=rows,
        path=path,
        fieldnames=["document_id", "text", "backend_name"],
    )


def save_classifications_csv(
    classifications: list[Classification],
    path: str | Path,
) -> None:
    """Save classifications to a CSV file."""
    rows = classifications_to_dicts(classifications)
    _save_csv(
        rows=rows,
        path=path,
        fieldnames=["document_id", "label", "rationale"],
    )


def _save_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    fieldnames: list[str],
) -> None:
    """Save dictionaries to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_csv_dict(
    value: Document | Response | Classification,
) -> dict[str, Any]:
    """Convert a docflow dataclass instance to a CSV-compatible dictionary."""
    row = asdict(value)

    for key, item in row.items():
        if isinstance(item, Path):
            row[key] = str(item)

    return row
