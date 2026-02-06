from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-small-en",
    local_files_only=True
)

# Cache loaded models so we don't reload every time
_loaded_models = {}

def embed(texts, model_name="BGE-small"):
    model_map = {
        "BGE-small": "BAAI/bge-small-en",
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "E5-small": "intfloat/e5-small-v2"
    }

    if model_name not in model_map:
        raise ValueError(f"Unsupported embedding model: {model_name}")

    hf_model_name = model_map[model_name]

    # Load once and reuse
    if hf_model_name not in _loaded_models:
        _loaded_models[hf_model_name] = SentenceTransformer(hf_model_name)

    model = _loaded_models[hf_model_name]

    return model.encode(texts)