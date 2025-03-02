import time
import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
# from langchain.chains import LLMChain
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
    print("✅ BM25 index created and saved.")

# Load Ollama LLM API
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
llm_models = {
    model: ChatOllama(model=model, base_url=OLLAMA_API_BASE_URL)
    for model in AVAILABLE_MODELS
}

def hybrid_search(query, top_k=3):
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
    chroma_results = retriever.get_relevant_documents(query)
    bm25_results = []
    
    if bm25:
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        bm25_results = [bm25_docs[i] for i in top_bm25_indices]

    combined_results = list({doc.page_content: doc for doc in chroma_results + bm25_results}.values())

    # Debugging output to verify retrieved documents
    print(f"🔍 Retrieved Documents:\n{[doc.page_content[:200] for doc in combined_results]}")

    return combined_results

total_sources = 2
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
    relevant_docs = hybrid_search(query, top_k=total_sources)

    # Construct context for the LLM
    context = "\n\n".join([
        f"**Section {doc.metadata.get('section', 'N/A')}** - {doc.metadata.get('title', 'Untitled')}\n{doc.page_content}"
        for doc in relevant_docs
    ])

    # **Using AIMessage & HumanMessage Format**
    messages = [
        SystemMessage(
            content="""
            You are a helpful assistant specializing in answering Home Owner Associlation rules. Respond in full sentences using natural language.
            Answer questions **ONLY** based on the provided HOA rules below.
            If a rule explicitly answers the question, state it **directly** with the rule number.
            If there is **no relevant rule**, say "There is no HOA rule covering this topic."
            
            📜 **HOA Rules for Reference:**
            """ + context
        ),
        HumanMessage(content=query)
    ]

    # **Initialize ChatOllama and invoke with messages**
    llm = ChatOllama(model=request.model, base_url=OLLAMA_API_BASE_URL)
    response = llm.invoke(messages)

    # Debugging: Print raw response
    print(f"🔍 Raw LLM Response: {response}")

    # **Handle structured response properly**
    if isinstance(response, AIMessage):
        llm_response = response.content  # Extract the actual AI-generated text
    elif isinstance(response, str):
        llm_response = response
    elif isinstance(response, dict):
        llm_response = response.get("content", "⚠️ No valid response received from LLM.")
    else:
        llm_response = "⚠️ Unexpected response format."

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
