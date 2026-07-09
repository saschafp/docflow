from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    id: str
    path: Path
    text: str


@dataclass
class Dataset:
    documents: list[Document]
    labels: dict[str, str]


@dataclass
class Response:
    document_id: str
    text: str
    backend_name: str


@dataclass
class Classification:
    document_id: str
    label: str
    rationale: str
