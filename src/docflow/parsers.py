from pathlib import Path
from typing import Protocol

from pypdf import PdfReader


class PdfParser(Protocol):
    def parse(self, path: Path) -> str: ...


class PyPdfParser:
    def parse(self, path: Path) -> str:
        reader = PdfReader(path)
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
