from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """
    A parsed input document.
    """

    id: str
    path: Path
    text: str


@dataclass
class Dataset:
    """
    A collection of documents with optinal true labels.
    """

    documents: list[Document]
    labels: dict[str, str] | None = None


@dataclass
class Response:
    """
    A response from a LLM backend for a given document.
    """

    document_id: str
    text: str
    backend_name: str


@dataclass
class Classification:
    """
    A parsed classification result for one document."""

    document_id: str
    label: str
    rationale: str
