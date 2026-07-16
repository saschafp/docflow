import docflow as df

dataset = df.dataset_from_folder("data/emails")

backend = df.LiteLLMBackend(model="ollama/gpt-oss:20b", url="http://localhost:11434")
allowed_labels = ["ham", "spam"]


system_prompt = df.PromptTemplate("""
You are a strict document classifier.
Return only valid JSON with the keys "label" and "rationale".
""")

user_prompt = df.PromptTemplate("""
Classify the following document.

Allowed labels:
{labels}

Document:
{document_text}
""").partial(labels=allowed_labels)


classifications = df.classify_documents(
    documents=dataset.documents,
    backend=backend,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    labels=allowed_labels,
)

matrix, matrix_labels = df.confusion_matrix(
    classifications=classifications,
    true_labels=dataset.labels,
)

df.plot_confusion_matrix(
    matrix=matrix,
    labels=matrix_labels,
)

for classification in classifications:
    print(f"Document ID: {classification.document_id}")
    print(f"Predicted Label: {classification.label}")
    print(f"Rationale: {classification.rationale}")
    print("-" * 40)