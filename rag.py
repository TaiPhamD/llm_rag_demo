#!/usr/bin/env python
# coding: utf-8



# Install required tools
# !pip install langchain langchain-community langchainhub llama-cpp-python
# !pip install pypdf chromadb tiktoken unstructured
# !pip install sentence-transformers




# Parse pdf documents into text
from langchain.document_loaders import PyPDFLoader

pdf_path = "/home/ppham/src/RAG/ThompsonWoodsCCRs.pdf"  # Replace with your file

# Load the document
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Combine all pages into a single string
raw_text = "\n".join([doc.page_content for doc in docs])




# We split text into smaller sections (500 tokens per chunk) so retrieval is efficient. Overlapping provides context.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create a text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)

# Split the document
documents = splitter.create_documents([raw_text])




# Since LLaMA 3.2 doesn’t natively support embeddings, we use OpenAI or Hugging Face embeddings.
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")




from langchain.vectorstores import Chroma

# Create a vector store
vector_store = Chroma.from_documents(documents, embedding_model)




# Download llm model
from huggingface_hub import hf_hub_download

# Define model repo and filename
model_repo = "unsloth/Llama-3.2-3B-Instruct-GGUF"  # Change if using a different model
model_filename = "Llama-3.2-3B-Instruct-F16.gguf"  # Choose a quantized version

# Download model
model_path = hf_hub_download(repo_id=model_repo, filename=model_filename, cache_dir="./models")

print("Model downloaded to:", model_path)




from langchain.llms import LlamaCpp

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=40,  # Use GPU acceleration
    n_batch=512,  # Increase batch size for better speed
    f16_kv=True,  # Use mixed precision
    temperature=0.2  # Reduce randomness
)




from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})  # Top 3 relevant chunks
)




query = "Can I paint my house blue?"
response = qa_chain.run(query)

print("🤖 Answer:", response)

