#!/usr/bin/env python
# coding: utf-8
# This server is the front-end to talk to ollama and provide openAI compatible endpoint 
# You can connect this server to openweb UI for example
# Requirement to use this server:
# 1. You have created embeddings and a vectorDB of your embeddings (see create_embeddings.py)
import time
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

import os

app = FastAPI()

# Define model name
MODEL_NAME = "llama3.2"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Load Ollama LLM
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_API_BASE_URL)

# Create retrieval-augmented generation (RAG) chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# OpenAI-compatible request format
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7

@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    if request.model != MODEL_NAME:
        return {"error": f"Model {request.model} not found"}, 404

    query = request.messages[-1]["content"]  # Extract last user message
    result = qa_chain(query)

    # Extract relevant sources
    source_documents = result.get("source_documents", [])

    sources = []
    formatted_sources = []
    
    for i, doc in enumerate(source_documents, 1):
        source_name = doc.metadata.get("source", "Unknown")  # PDF file name
        page_number = doc.metadata.get("page", "Unknown")  # Page number
        snippet = doc.page_content[:200]  # First 200 chars of content
        
        source_info = {
            "source": source_name,
            "page": page_number,
            "snippet": snippet
        }
        sources.append(source_info)

        # Enhance readability with better formatting
        formatted_sources.append(f"📄 **Source {i}:** *{source_name}* (Page {page_number})\n🔹 _Snippet:_ {snippet}...\n")

    # OpenAI-compatible response format
    response = {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"{result['result']}\n\n**Sources:**\n" + "\n".join(formatted_sources)
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 50,  # Placeholder, can be calculated if needed
            "completion_tokens": 100,
            "total_tokens": 150
        },
        "source_documents": sources  # Attach sources to the response
    }

    return response



@app.get("/v1/models")
def list_models():
    """Endpoint for OpenWebUI to retrieve available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "custom",
                "permission": []
            }
        ]
    }
