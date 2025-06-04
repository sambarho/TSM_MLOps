'''
embeddings.py

This module provides functionality to generate sentence embeddings using either:
- A remote Hugging Face Inference API (when USE_REMOTE is True)
- A locally cached SentenceTransformer model (when USE_REMOTE is False)

It automatically handles downloading the model snapshot if it is not already present locally.
'''
import os
from ollama import Client as OllamaClient
import openai
from huggingface_hub import InferenceApi, snapshot_download
from sentence_transformers import SentenceTransformer
from pathlib import Path

USE_REMOTE = os.getenv("USE_REMOTE","false").lower()=="true"
openai.api_key = os.getenv("OPENAI_API_KEY")
hf = InferenceApi("sentence-transformers/all-MiniLM-L6-v2", task="feature-extraction",
                  token=os.getenv("HF_API_TOKEN"))
_local_model = None

def embed(sentences: list[str]) -> list[list[float]]:
    '''
    Generate embeddings for a list of sentences.

    If USE_REMOTE is True, the function calls the Hugging Face Inference API to compute embeddings.
    Otherwise, it lazily loads a local copy of the SentenceTransformer model (all-MiniLM-L6-v2)
    from a 'models' directory one level up, downloads it if necessary, and uses it to compute embeddings.

    Parameters:
        sentences (list[str]): A list of input strings for which embeddings are to be computed.

    Returns:
        list[list[float]]: A list of embedding vectors (one list of floats per input sentence).
    '''
    if USE_REMOTE:
        return hf(sentences)
    else:
        global _local_model
        if _local_model is None:
            root = Path(__file__).parent.parent / "models"
            local = root / "all-MiniLM-L6-v2"
            if not local.exists():
                local.mkdir(parents=True,exist_ok=True)
                path = snapshot_download("sentence-transformers/all-MiniLM-L6-v2",
                                         cache_dir=str(root))
            else:
                path = str(local)
            _local_model = SentenceTransformer(path)
        return _local_model.encode(sentences, convert_to_tensor=False).tolist()
