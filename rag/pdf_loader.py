"""
pdf_loader.py
-------------
Extracts plain text from PDF files using PyPDF.
"""

import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_pdf(path: str) -> str:
    """
    Extract and concatenate text from all pages of a PDF file.

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        Full extracted text as a single string.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        ValueError:        If the PDF contains no extractable text.
    """
    logger.info("Loading PDF: %s", path)

    reader = PdfReader(path)
    pages = reader.pages

    if not pages:
        raise ValueError(f"PDF has no pages: {path}")

    text = ""
    for page in pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        raise ValueError(
            f"No extractable text found in PDF: {path}. "
            f"The file may be scanned or image-based."
        )

    logger.info("Extracted %d characters from %d pages.", len(text), len(pages))
    return text