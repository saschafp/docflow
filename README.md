# docflow

`docflow` is a small Python package for running collections of documents through LLM backends.

## Installation
From the repository root, run:
```bash
pip install -e .
```

If you want to use the optional progress bar, install `tqdm`:
```bash
pip install tqdm
```
or install it with the progress dependencies:
```bash
pip install -e .[progress]
```

For development, install the dev dependencies:
```bash
pip install -e .[dev]
```

## Example
Run the [email classification example](examples/classification.py):
```bash
python examples/classification.py
```

The example assumes Ollama is running locally and `gpt-oss:20b` is available.
