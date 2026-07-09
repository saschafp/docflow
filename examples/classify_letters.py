from docflow.backends import LiteLLMBackend
from docflow.classification import classifications_from_responses
from docflow.documents import documents_from_folder
from docflow.generation import generate_responses
from docflow.labels import labels_from_csv
from docflow.metrics import confusion_matrix, plot_confusion_matrix
from docflow.prompts import PromptTemplate

docs = documents_from_folder("data/letters")
true_labels = labels_from_csv("data/letters/labels.csv")

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


responses = generate_responses(
    documents=docs,
    backend=backend,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
)

classifications = classifications_from_responses(
    responses=responses,
    labels=allowed_labels,
)

matrix, matrix_labels = confusion_matrix(
    classifications=classifications,
    true_labels=true_labels,
)

plot_confusion_matrix(
    matrix=matrix,
    labels=matrix_labels,
)
