from docflow.backends import LiteLLMBackend
from docflow.classification import classify
from docflow.documents import documents_from_folder
from docflow.metrics import confusion_matrix, plot_confusion_matrix

docs = documents_from_folder("data/letters")

backend = LiteLLMBackend(model="ollama/mistral:latest", url="http://localhost:11434")

labels = ["A", "B", "C"]

predictions = classify(documents=docs, backend=backend, labels=labels)

true_labels = {"A": "A", "B": "B", "C": "C"}
matrix, labels = confusion_matrix(predictions=predictions, true_labels=true_labels)
plot_confusion_matrix(matrix=matrix, labels=labels, output_path=None)

for prediction in predictions:
    print(prediction)
