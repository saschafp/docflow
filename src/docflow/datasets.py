from pathlib import Path

from .documents import documents_from_folder
from .labels import labels_from_csv
from .schemas import Dataset


def dataset_from_folder(
    path: Path | str,
    labels_file: str = "labels.csv",
) -> Dataset:
    path = Path(path)

    documents = documents_from_folder(path)

    labels_path = path / labels_file
    labels = labels_from_csv(labels_path) if labels_path.exists() else None

    return Dataset(
        documents=documents,
        labels=labels,
    )
