#!/usr/bin/env bash
set -e

echo ">>> Starting Ollama…" >&2
ollama serve > /tmp/ollama.log 2>&1 &

echo ">>> Waiting for Ollama API…" >&2
until curl -s http://localhost:11434 > /dev/null; do
     echo "…still waiting" >&2
     sleep 1
done
echo ">>> Ollama is up!" >&2

echo ">>> Pulling Mistral model…" >&2
ollama pull mistral >> /tmp/ollama.log 2>&1

echo ">>> Ollama ready. Launching Streamlit…" >&2
exec streamlit run app/main.py \
     --server.port $PORT \
     --server.address 0.0.0.0 \
     --global.developmentMode=false
echo ">>> Streamlit started." >&2
echo ">>> Ollama log:" >&2