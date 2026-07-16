from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

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
    progress=True,
)

# Print the classification results
for classification in classifications:
    print(f"Document ID: {classification.document_id}")
    print(f"Predicted Label: {classification.label}")
    print(f"Rationale: {classification.rationale}")
    print("-" * 40)


# Save the classification results
df.save_classifications_csv(
    classifications=classifications, path="out/classification_results.csv"
)

# Evaluate the classification results
y_true, y_pred = df.classification_targets(
    classifications=classifications, true_labels=dataset.labels
)

# Sklearn confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=allowed_labels)

# Plot with matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = range(len(allowed_labels))
plt.xticks(tick_marks, allowed_labels)
plt.yticks(tick_marks, allowed_labels)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
