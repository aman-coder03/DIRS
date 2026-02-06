import os
import csv
from datetime import datetime

LOG_FILE = "results/llm_benchmark.csv"

def log_experiment(document, llm, embedding, vector_db, metrics):

    os.makedirs("results", exist_ok=True)

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

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
                "answer_length_chars"
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
            metrics["answer_length_chars"]
        ])
