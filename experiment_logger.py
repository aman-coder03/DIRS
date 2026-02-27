"""
experiment_logger.py
--------------------
Appends query performance metrics to a CSV log file for benchmarking.
"""

import os
import csv
import logging
from datetime import datetime

from config import RESULTS_FILE

logger = logging.getLogger(__name__)


def log_experiment(
    document: str,
    llm: str,
    embedding: str,
    vector_db: str,
    metrics: dict,
) -> None:
    """
    Append a single experiment result row to the benchmark CSV.

    Creates the results directory and writes a header row automatically
    if the file does not yet exist.

    Args:
        document:   Name of the indexed document queried.
        llm:        LLM model used for generation.
        embedding:  Embedding model used for retrieval.
        vector_db:  Vector database used (FAISS or Chroma).
        metrics:    Dictionary of performance metrics returned by query_index().
    """
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    file_exists = os.path.isfile(RESULTS_FILE)

    try:
        with open(RESULTS_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "document",
                    "llm",
                    "embedding_model",
                    "vector_db",
                    "embedding_time",
                    "retrieval_time",
                    "generation_time",
                    "total_time",
                    "tokens_per_second",
                    "prompt_length_chars",
                    "answer_length_chars",
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                document,
                llm,
                embedding,
                vector_db,
                metrics["embedding_time"],
                metrics["retrieval_time"],
                metrics["generation_time"],
                metrics["total_time"],
                metrics["tokens_per_second"],
                metrics["prompt_length_chars"],
                metrics["answer_length_chars"],
            ])

        logger.info("Experiment logged to %s.", RESULTS_FILE)

    except OSError as e:
        logger.warning("Failed to write experiment log: %s", e)