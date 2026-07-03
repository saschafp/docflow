from docflow.backends import LiteLLMBackend
from docflow.classification import classify
from docflow.documents import documents_from_folder

docs = documents_from_folder("data/letters")

backend = LiteLLMBackend(
    model="ollama/mistral:latest",
    url="http://localhost:11434"
)

labels = ["A", "B", "C"]

predictions = classify(documents=docs, backend=backend, labels=labels)

for prediction in predictions:
    print(prediction)
