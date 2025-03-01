# Pre-requisites

1. Docker setup
1. (optional) CUDA toolkit / nvidia GPU for speed. If you don't have nvidia GPU then will need to edit the provided docker-compose.yml and remove the `deploy` [section](https://github.com/TaiPhamD/llm_rag_demo/blob/cd6661f4720b3546962d93c33792ad0b43c9f20d/docker-compose.yml#L38)

# How to use

1. Create your python venv with your favorite tool conda or e.g. `python3 -m venv .venv` 
1. Install ptyhon depedendencies `pip install -r requirements.txt`
1. Create embeddings/vectorDB of your document run `./create_embeddings.py your_file.pdf`
1. Start server via `docker compose up --build -d`. See the `docker-compose.yml` to see how ollama, rag-api, and openwebUI are used together. 
  - Use `docker compose down` if you want to shut down the service.
1. Visit localhost:8080 to login into openwebui
