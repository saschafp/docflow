import csv
from pathlib import Path


def labels_from_csv(path: str | Path) -> dict[str, str]:
    path = Path(path)

    with path.open("r") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"No fieldnames found in CSV file: {path}")

        required_columns = {"document_id", "label"}
        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(f"Missing required columns in CSV file: {missing_columns}")

        return {row["document_id"]: row["label"] for row in reader}
