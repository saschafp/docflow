from docflow.backends import DummyBackend
from docflow.classification import classify
from docflow.documents import documents_from_folder

docs = documents_from_folder("data/letters")
backend = DummyBackend()
labels = ["A", "B", "C"]

predictions = classify(documents=docs, backend=backend, labels=labels)

for prediction in predictions:
    print(prediction)
