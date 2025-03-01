# Pre-requisites

1. Docker setup
1. (optional) CUDA toolkit / nvidia GPU for speed

# How to use

1. Create your python venv with your favorite tool conda or e.g. `python3 -m venv .venv` 
1. Install ptyhon depedendencies `pip install -r requirements.txt`
1. Create embeddings/vectorDB of your document run `./create_embeddings.py your_file.pdf`
1. Start server via `docker compose up --build -d`. See the `docker-compose.yml` to see how ollama, rag-api, and openwebUI are used together. 
  - Use `docker compose down` if you want to shut down the service.
1. Visit localhost:8080 to login into openwebui
