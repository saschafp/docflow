from pathlib import Path
import matplotlib.pyplot as plt

from .schemas import Classification


def confusion_matrix(
    classifications: list[Classification],
    true_labels: dict[str, str],
) -> tuple[list[list[int]], list[str]]:
    """
    Compute the confusion matrix from a classification results and true labels.
    """
    y_true: list[str] = []
    y_pred: list[str] = []

    for classification in classifications:
        y_true.append(true_labels[classification.document_id])
        y_pred.append(classification.label)

    labels = sorted(set(y_true) | set(y_pred))
    label_to_index = {label: i for i, label in enumerate(labels)}

    matrix = [[0 for _ in labels] for _ in labels]

    for true, pred in zip(y_true, y_pred):
        matrix[label_to_index[true]][label_to_index[pred]] += 1

    return matrix, labels


def plot_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    output_path: Path | None = None,
) -> None:
    """
    Plot or save the confusion matrix.
    """
    fig, ax = plt.subplots()

    ax.imshow(matrix)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="white")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    else:
        plt.show()
