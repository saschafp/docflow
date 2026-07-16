from .schemas import Classification


def classification_targets(
    classifications: list[Classification],
    true_labels: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Return true and predicted labels for classification evaluation."""
    y_true: list[str] = []
    y_pred: list[str] = []

    for classification in classifications:
        y_true.append(true_labels[classification.document_id])
        y_pred.append(classification.label)

    return y_true, y_pred
