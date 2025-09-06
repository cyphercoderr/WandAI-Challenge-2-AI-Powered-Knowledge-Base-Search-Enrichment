import os
import numpy as np
try:
    import openai
except Exception:
    openai = None

USE_OPENAI = bool(os.getenv("OPENAI_API_KEY")) and openai is not None

def embed_texts(texts):
    """Return list of embeddings for given texts.
    Uses OpenAI embeddings when OPENAI_API_KEY present, otherwise falls back to TF-IDF embeddings (handled elsewhere).
    """
    if USE_OPENAI:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        resp = openai.Embedding.create(model="text-embedding-3-small", input=texts)
        return [r["embedding"] for r in resp["data"]]
    else:
        raise RuntimeError("OpenAI embeddings not enabled")
