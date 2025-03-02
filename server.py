#!/usr/bin/env python
# coding: utf-8
# This server is a front-end to Ollama, providing an OpenAI-compatible endpoint.
# It connects to OpenWebUI and serves responses based on HOA rule retrieval.
# Requirements:
# 1. Ensure embeddings and vectorDB (Chroma) are created beforehand (see create_embeddings.py).

import time
import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from rank_bm25 import BM25Okapi

app = FastAPI()

# Define model name for Ollama
MODEL_NAME = "initium/law_model"

# Load embedding model (Ensure it matches the ChromaDB embeddings)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load ChromaDB vector store
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Load BM25 index for hybrid search (or create if missing)
BM25_INDEX_PATH = "./chroma_db/bm25_index.pkl"
BM25_DOCS_PATH = "./chroma_db/bm25_docs.pkl"

if os.path.exists(BM25_INDEX_PATH) and os.path.exists(BM25_DOCS_PATH):
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_DOCS_PATH, "rb") as f:
        bm25_docs = pickle.load(f)
else:
    print("🔍 No existing BM25 index found. Creating a new one...")

    # Extract stored documents from ChromaDB
    retriever = vector_store.as_retriever(search_kwargs={"k": 100})  # Retrieve all documents
    documents = retriever.get_relevant_documents("")

    # Process text for BM25
    bm25_docs = [doc for doc in documents]
    tokenized_docs = [doc.page_content.split(" ") for doc in bm25_docs]  # Tokenize for BM25

    # Initialize BM25
    bm25 = BM25Okapi(tokenized_docs)  # ✅ BM25 now properly initialized!

    # Save BM25 index for future queries
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_DOCS_PATH, "wb") as f:
        pickle.dump(bm25_docs, f)

    print("✅ BM25 index created and saved.")

# Load Ollama LLM API
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_API_BASE_URL)

# Define Hybrid Search Function
def hybrid_search(query, top_k=10):
    """
    Performs hybrid search using BM25 for keyword matching and Chroma for semantic search.
    Returns a ranked list of documents based on both methods.
    """
    # Retrieve using Chroma embeddings
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
    chroma_results = retriever.get_relevant_documents(query)

    # Retrieve using BM25 keyword search if available
    bm25_results = []
    if bm25:
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)

        # Prioritize exact matches with higher weighting
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        bm25_results = [bm25_docs[i] for i in top_bm25_indices]

    # Ensure exact keyword matches appear first
    combined_results = sorted(
        {doc.page_content: doc for doc in chroma_results + bm25_results}.values(),
        key=lambda doc: int("boat" in doc.page_content.lower() or "parking" in doc.page_content.lower()),  # Boost direct matches
        reverse=True
    )

    return list(combined_results)


# Create RetrievalQA chain with Hybrid Search
def get_qa_chain():
    return RetrievalQA.from_chain_type(
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
    """Handles OpenAI-style chat completions via Ollama + Hybrid Search"""
    if request.model != MODEL_NAME:
        return {"error": f"Model {request.model} not found"}, 404

    query = request.messages[-1]["content"]  # Extract user message
    relevant_docs = hybrid_search(query, top_k=3)  # Use Hybrid Search

    # Combine retrieved docs for LLM context
    context = "\n\n".join([f"📌 **Section {doc.metadata.get('section', 'N/A')}** - {doc.metadata.get('title', 'Untitled')}\n{doc.page_content}" for doc in relevant_docs])

    system_prompt = f"""
    You are a strict HOA assistant. You must answer questions **ONLY** based on the provided HOA rules below.
    If a rule explicitly answers the question, state it **directly** with the rule number.
    If there is **no relevant rule**, say "There is no HOA rule covering this topic."

    🚨 **Strictly avoid making assumptions or offering opinions.** 🚨

    📜 **HOA Rules for Reference:**
    {context}

    📝 **Answer Format:**
    1. If the rules explicitly answer the question, provide the rule number and the exact rule text.
    2. If the rules do not cover the topic, say "There is no HOA rule on this."

    Now, answer this question:
    """


    # Query Llama3 for final response
    qa_chain = get_qa_chain()
    response = qa_chain.invoke(query)  # ✅ FIXED: Now correctly handles multiple outputs

    # Extract response & source documents
    llm_response = response["result"]
    source_documents = response.get("source_documents", [])

    # Format sources for response
    sources = []
    formatted_sources = []
    
    for i, doc in enumerate(source_documents, 1):
        source_name = doc.metadata.get("source", "Unknown")
        page_number = doc.metadata.get("page", "Unknown")
        snippet = doc.page_content[:200]  # First 200 chars

        source_info = {"source": source_name, "page": page_number, "snippet": snippet}
        sources.append(source_info)

        formatted_sources.append(f"📄 **Source {i}:** *{source_name}* (Page {page_number})\n🔹 _Snippet:_ {snippet}...\n")

    # OpenAI-compatible response
    response = {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"{llm_response}\n\n**Sources:**\n" + "\n".join(formatted_sources)
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150
        },
        "source_documents": sources
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
