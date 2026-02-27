"""
chunker.py
----------
Splits extracted document text into overlapping chunks for retrieval.
"""

import logging

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into fixed-size overlapping chunks.

    Args:
        text:       The full document text to split.
        chunk_size: Number of characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        List of text chunk strings.

    Raises:
        ValueError: If chunk_size <= overlap.
    """
    if chunk_size <= overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})."
        )

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    logger.info("Created %d chunks (size=%d, overlap=%d).", len(chunks), chunk_size, overlap)
    return chunks