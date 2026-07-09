# docflow

`docflow` is a small Python package for running collections of documents through LLM backends.

## Installation
From the repository root, run:
```bash
pip install -e .
```

## Example
Run the [letter classification example](examples/classify_letters.py):
```bash
python examples/classify_letters.py
```

The example assumes Ollama is running locally and `mistral:latest` is available.
