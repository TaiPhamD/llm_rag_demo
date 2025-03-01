# Pre-requisites

1. Docker setup
1. Ollama installed and pulled llama3.2 model
1. (optional) CUDA toolkit / nvidia GPU for speed

# How to use

1. Create your python venv with your favorite tool conda or e.g. `python3 -m venv .venv` 
1. Install ptyhon depedendencies `pip install -r requirements.txt`
1. Create embeddings/vectorDB of your document run `python create_embeddings.py`
1. Start server via `docker compose up -d build`
1. Visit localhost:8080 to login into openwebui
