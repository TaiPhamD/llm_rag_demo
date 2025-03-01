#!/usr/bin/env python
# coding: utf-8

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Set paths
pdf_path = "/home/ppham/src/RAG/ThompsonWoodsCCRs.pdf"  # Update this with your actual PDF path
persist_directory = "./chroma_db"  # Where to store the embeddings

print("🔍 Creating embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 1: Load PDF and Extract Text with Metadata
print("📄 Loading PDF...")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Extract raw text and maintain metadata
documents = []
for i, doc in enumerate(docs):
    metadata = {
        "source": pdf_path,  # Store the PDF file name
        "page": i + 1  # Store the page number (starting from 1)
    }
    documents.append({"content": doc.page_content, "metadata": metadata})

# Step 2: Split the Text into Chunks and Preserve Metadata
print("✂️ Splitting text into smaller chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

split_docs = []
for doc in documents:
    chunks = splitter.split_text(doc["content"])
    for chunk in chunks:
        split_docs.append({
            "content": chunk,
            "metadata": doc["metadata"]  # Preserve metadata across chunks
        })

# Step 3: Convert into Chroma-compatible Format
from langchain.schema import Document

chroma_documents = [
    Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in split_docs
]

# Step 4: Store the Embeddings in ChromaDB
print(f"💾 Saving embeddings to {persist_directory} ...")
vector_store = Chroma.from_documents(chroma_documents, embedding_model, persist_directory=persist_directory)

# Ensure the database is saved
vector_store.persist()
print("✅ Embeddings successfully created and saved!")
