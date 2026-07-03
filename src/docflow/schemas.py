from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    id: str
    path: Path
    text: str


@dataclass
class Prediction:
    document_id: str
    label: str
    rationale: str
