from docflow.backends import LiteLLMBackend
from docflow.classification import classifications_from_responses
from docflow.documents import documents_from_folder
from docflow.generation import generate_responses
from docflow.prompts import PromptTemplate


docs = documents_from_folder("data/letters")
labels = ["A", "B", "C"]
backend = LiteLLMBackend(model="ollama/mistral:latest", url="http://localhost:11434")


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
""").partial(labels=labels)


responses = generate_responses(
    documents=docs,
    backend=backend,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
)

classifications = classifications_from_responses(
    responses=responses,
    labels=labels,
)

print("Responses:")
for response in responses:
    print(response)

print("\nClassifications:")
for classification in classifications:
    print(classification)
