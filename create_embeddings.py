#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

def parse_args():
    parser = argparse.ArgumentParser(description="Process a PDF and create embeddings using ChromaDB.")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file.")
    parser.add_argument("--persist-dir", type=str, default="./chroma_db", help="Directory to store embeddings.")
    return parser.parse_args()

def main():
    args = parse_args()
    pdf_path = args.pdf_path
    persist_directory = args.persist_dir

    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file '{pdf_path}' not found.")
        return

    print("🔍 Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 1: Load PDF and Extract Text with Metadata
    print("📄 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Extract raw text and maintain metadata
    print("📑 Extracting text from PDF pages...")
    documents = []
    with tqdm(total=len(docs), desc="Processing pages", unit="page") as pbar:
        for i, doc in enumerate(docs):
            metadata = {"source": pdf_path, "page": i + 1}
            documents.append({"content": doc.page_content, "metadata": metadata})
            pbar.update(1)

    # Step 2: Split the Text into Chunks and Preserve Metadata
    print("✂️ Splitting text into smaller chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    split_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc["content"])
        for chunk in chunks:
            split_docs.append({"content": chunk, "metadata": doc["metadata"]})

    # Step 3: Convert into Chroma-compatible Format
    chroma_documents = [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in split_docs]

    # Step 4: Store the Embeddings in ChromaDB with Progress Bar
    print(f"💾 Saving embeddings to {persist_directory} ...")
    with tqdm(total=len(chroma_documents), desc="Embedding progress", unit="doc") as pbar:
        vector_store = Chroma.from_documents(chroma_documents, embedding_model, persist_directory=persist_directory)
        pbar.update(len(chroma_documents))
    
    print("✅ Embeddings successfully created and saved!")

if __name__ == "__main__":
    main()
