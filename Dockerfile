# Dockerfile

# 1) Start from a slim Python image
FROM python:3.11-slim

# 2) Install system dependencies (git-lfs for HF models, curl for Ollama installer)
RUN apt-get update && \
    apt-get install -y build-essential git-lfs curl && \
    rm -rf /var/lib/apt/lists/*

# 3) Install Ollama CLI (for your local Mistral LLM)
RUN curl -fsSL https://ollama.com/install.sh | sh

# 4) Set working directory
WORKDIR /app

# 5) Copy and install Python requirements
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt


# 6) Copy your application code
COPY resume_matcher/app/ ./app
COPY resume_matcher/data/ ./data

# 7) Pre‐download the MiniLM model into ./models (optional; speeds first startup)
RUN python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("sentence-transformers/all-MiniLM-L6-v2", cache_dir="./models")
EOF

# 8) Expose Streamlit’s and Ollama’s ports
# (8501 for Streamlit, 11434 for Ollama)
EXPOSE 8501 11434

COPY start.sh /start.sh
RUN chmod +x /start.sh

# 9) When container starts: launch Ollama, then Streamlit
#ENTRYPOINT ["bash","-lc","ollama serve & sleep 8 && exec streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0"]
ENTRYPOINT ["/start.sh"]
