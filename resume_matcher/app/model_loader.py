'''
model_loader.py

This module provides a cached function to load the SentenceTransformer model
either from a local directory or by downloading it from Hugging Face.
It uses Streamlit's caching to avoid re-downloading on each run.
'''
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import streamlit as st

models_root = Path("..") / "models"

@st.cache_resource
def load_model():
    '''
    Load the 'all-MiniLM-L6-v2' SentenceTransformer model, using caching to speed up repeated calls.

    Behavior:
        1. Check if a local copy of the model exists under models_root/all-MiniLM-L6-v2.
           - If it exists, set model_path to that directory.
           - If not, create the models_root directory if needed and download the model snapshot.
        2. Return a SentenceTransformer instance loaded from model_path.

    Returns:
        SentenceTransformer: The loaded model instance.
    '''
    local_subdir = models_root / "all-MiniLM-L6-v2"

    if local_subdir.is_dir():
        # 1) already downloaded
        model_path = str(local_subdir)
    else:
        # 2) pull from HF and get back the true download path
        models_root.mkdir(parents=True, exist_ok=True)
        model_path = snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=str(models_root),
            repo_type="model"
        )

    #load from disk (or from the freshly-downloaded folder)
    return SentenceTransformer(model_path)