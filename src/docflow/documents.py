from pathlib import Path

from .parsers import PdfParser, PyPdfParser
from .schemas import Document


def document_from_pdf(
    path: str | Path,
    parser: PdfParser | None = None,
) -> Document:
    """
    Load a PDF file as a Document.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {path}")

    if parser is None:
        parser = PyPdfParser()

    text = parser.parse(path)

    return Document(
        id=path.name,
        path=path,
        text=text,
    )


def documents_from_folder(
    path: str | Path,
    parser: PdfParser | None = None,
) -> list[Document]:
    """
    Load all PDF files in a folder as Documents.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Folder not found: {path}")

    if not path.is_dir():
        raise ValueError(f"Path is not a folder: {path}")

    pdf_paths = sorted(path.glob("*.pdf"))

    return [document_from_pdf(pdf_path, parser=parser) for pdf_path in pdf_paths]
