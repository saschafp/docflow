from pathlib import Path
from typing import Protocol

from pypdf import PdfReader


class PdfParser(Protocol):
    """
    Interface for PDF-to-text parsers.
    """

    def parse(self, path: Path) -> str: ...


class PyPdfParser:
    """
    PDF parser based on pypdf.
    """

    def parse(self, path: Path) -> str:
        """
        Extract text from a PDF file using pypdf.
        """
        reader = PdfReader(path)
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
