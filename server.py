import time
import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi

app = FastAPI()

# Define available models
AVAILABLE_MODELS = ["llama3.2", "llama3.1", "initium/law_model"]

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load ChromaDB vector store
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Load BM25 index
BM25_INDEX_PATH = "./chroma_db/bm25_index.pkl"
BM25_DOCS_PATH = "./chroma_db/bm25_docs.pkl"

if os.path.exists(BM25_INDEX_PATH) and os.path.exists(BM25_DOCS_PATH):
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_DOCS_PATH, "rb") as f:
        bm25_docs = pickle.load(f)
else:
    print("🔍 No existing BM25 index found. Creating a new one...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 100})
    documents = retriever.get_relevant_documents("")
    bm25_docs = [doc for doc in documents]
    tokenized_docs = [doc.page_content.split(" ") for doc in bm25_docs]
    bm25 = BM25Okapi(tokenized_docs)
    
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_DOCS_PATH, "wb") as f:
        pickle.dump(bm25_docs, f)

    print(f"✅ BM25 index created with {len(bm25_docs)} documents.")

def hybrid_search(query, top_k=5):
    """Hybrid search using ChromaDB & BM25."""
    
    # ChromaDB retrieval (using similarity instead of mmr)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    chroma_results = retriever.get_relevant_documents(query)

    # BM25 retrieval
    bm25_results = []
    if bm25:
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        bm25_results = [bm25_docs[i] for i in top_bm25_indices]

    # Merge results while keeping unique documents
    merged_results = {}
    for doc in chroma_results + bm25_results:
        key = f"{doc.metadata.get('section', 'N/A')}-{doc.metadata.get('page', 'Unknown')}"
        merged_results[key] = doc  # Overwrite duplicates

    final_results = list(merged_results.values())[:top_k]

    print(f"🔍 Retrieved {len(final_results)} documents:")
    for doc in final_results:
        print(f"📄 Section: {doc.metadata.get('section', 'N/A')} - Title: {doc.metadata.get('title', 'Untitled')}\n{doc.page_content[:200]}...\n")

    return final_results

# OpenAI-compatible request format
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7

@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    if request.model not in AVAILABLE_MODELS:
        return {"error": f"Model {request.model} not found"}, 404

    query = request.messages[-1]["content"]

    # Retrieve relevant documents
    relevant_docs = hybrid_search(query, top_k=5)

    # Construct context for the LLM
    context = "\n\n".join([
        f"**Section {doc.metadata.get('section', 'N/A')}** - {doc.metadata.get('title', 'Untitled')}\n{doc.page_content[:1000]}"
        for doc in relevant_docs
    ])

    # Debugging LLM input
    print(f"🔍 Context for LLM:\n{context[:2000]}...\n")

    # Format messages for LLM
    messages = [
        SystemMessage(
            content=f"""
            You are a helpful assistant specializing in answering Homeowner Association rules. 
            Answer questions **ONLY** based on the HOA rules below.
            
            📜 **HOA Rules for Reference:**
            {context}
            """
        ),
        HumanMessage(content=query)
    ]

    # Initialize ChatOllama and invoke
    llm = ChatOllama(model=request.model, base_url=os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434"))
    response = llm.invoke(messages)

    # Debugging LLM response
    print(f"🔍 Raw LLM Response: {response}")

    # Handle structured response
    llm_response = response.content if isinstance(response, AIMessage) else response

    # Format sources
    sources = []
    formatted_sources = []
    for i, doc in enumerate(relevant_docs, 1):
        source_name = doc.metadata.get("source", "Unknown")
        page_number = doc.metadata.get("page", "Unknown")
        snippet = doc.page_content[:200]

        source_info = {"source": source_name, "page": page_number, "snippet": snippet}
        sources.append(source_info)

        formatted_sources.append(f"📄 **Source {i}:** *{source_name}* (Page {page_number})\n🔹 _Snippet:_ {snippet}...\n")

    return {
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

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "custom",
                "permission": []
            }
            for model in AVAILABLE_MODELS
        ]
    }
