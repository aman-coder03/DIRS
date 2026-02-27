"""
llm.py
------
Interfaces with a locally running Ollama server to generate answers.
"""

import logging
import ollama

from config import MAX_NEW_TOKENS

logger = logging.getLogger(__name__)


def generate_answer(prompt: str, model_name: str = "llama3:latest") -> str:
    """
    Send a prompt to the specified Ollama model and return the generated text.

    Args:
        prompt:     The full prompt string to send to the model.
        model_name: Ollama model identifier (e.g. 'llama3:latest').

    Returns:
        Generated answer as a string.

    Raises:
        RuntimeError: If the Ollama server is unreachable or the model is not found.
    """
    logger.info("Generating answer with model: %s", model_name)

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": MAX_NEW_TOKENS},
        )
        return response["message"]["content"]

    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        raise RuntimeError(
            f"Could not generate answer using '{model_name}'. "
            f"Ensure Ollama is running and the model has been pulled.\n"
            f"Details: {e}"
        ) from e