from docflow.backends import LiteLLMBackend
from docflow.classification import classify_documents
from docflow.datasets import dataset_from_folder
from docflow.metrics import confusion_matrix, plot_confusion_matrix
from docflow.prompts import PromptTemplate

dataset = dataset_from_folder("data/letters")

backend = LiteLLMBackend(model="ollama/mistral:latest", url="http://localhost:11434")
allowed_labels = ["A", "B", "C"]


system_prompt = PromptTemplate("""
You are a strict document classifier.
Return only valid JSON with the keys "label" and "rationale".
""")

user_prompt = PromptTemplate("""
Classify the following document.

Allowed labels:
{labels}

Document:
{document_text}
""").partial(labels=allowed_labels)


classifications = classify_documents(
    documents=dataset.documents,
    backend=backend,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    labels=allowed_labels,
)

matrix, matrix_labels = confusion_matrix(
    classifications=classifications,
    true_labels=dataset.labels,
)

plot_confusion_matrix(
    matrix=matrix,
    labels=matrix_labels,
)
