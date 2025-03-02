#!/usr/bin/env python
# coding: utf-8

import os
import re
import argparse
import pickle
import pdfplumber
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from rank_bm25 import BM25Okapi

def parse_args():
    parser = argparse.ArgumentParser(description="Process a PDF and create embeddings using ChromaDB.")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file.")
    parser.add_argument("--persist-dir", type=str, default="./chroma_db", help="Directory to store embeddings.")
    return parser.parse_args()

def extract_toc_from_pdf(pdf_path):
    """
    Dynamically extracts the Table of Contents (TOC) from a PDF file.
    Returns a dictionary mapping section numbers to titles and page numbers.
    """
    toc_map = {}
    toc_detected = False
    pattern = r'(\d+\.\d+)\s+"([^"]+)"(?:\s+or\s+"[^"]+")?\s+\.*\s+(\d+)'

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue  # Skip empty pages

            if not toc_detected and "ARTICLE" in text.upper():
                toc_detected = True  

            if toc_detected:
                matches = re.findall(pattern, text)
                if not matches:
                    break  # Stop at first non-TOC page

                for match in matches:
                    section_number, title, page_number = match[0], match[1], int(match[2])
                    toc_map[section_number] = {"title": title, "page": page_number}

    return toc_map

def extract_rules(text, toc_map):
    """
    Uses TOC data to extract structured HOA rules.
    Returns a list of dictionaries containing section number, title, and body.
    """
    sections = []
    current_section = None
    section_text = []

    lines = text.split("\n")
    for line in lines:
        match = re.match(r"(\d+\.\d+)\s+(.*)", line.strip())
        if match:
            section_number, section_title = match.groups()

            if current_section:
                sections.append({
                    "section": current_section,
                    "title": toc_map.get(current_section, {}).get("title", "Unknown"),
                    "content": " ".join(section_text)
                })
                section_text = []

            current_section = section_number  

        section_text.append(line.strip())

    if current_section:
        sections.append({
            "section": current_section,
            "title": toc_map.get(current_section, {}).get("title", "Unknown"),
            "content": " ".join(section_text)
        })

    return sections

def main():
    args = parse_args()
    pdf_path = args.pdf_path
    persist_directory = args.persist_dir
    print(persist_directory)

    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file '{pdf_path}' not found.")
        return

    print("🔍 Extracting TOC dynamically...")
    toc_map = extract_toc_from_pdf(pdf_path)

    print("📄 Loading full text from PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print("📑 Extracting text from PDF pages...")
    full_text = "\n".join([doc.page_content for doc in docs])
    structured_rules = extract_rules(full_text, toc_map)

    chroma_documents = [
        Document(page_content=rule["content"], metadata={"section": rule["section"], "title": rule["title"]})
        for rule in structured_rules
    ]

    print("🔍 Creating embeddings using Legal-BERT...")
    embedding_model = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")

    print(f"💾 Saving embeddings to {persist_directory} ...")
    with tqdm(total=len(chroma_documents), desc="Embedding progress", unit="doc") as pbar:
        vector_store = Chroma.from_documents(chroma_documents, embedding_model, persist_directory=persist_directory)
        pbar.update(len(chroma_documents))

    print("🔍 Indexing for Hybrid Search (BM25 + Embeddings)...")
    tokenized_docs = [doc.page_content.split(" ") for doc in chroma_documents]
    bm25 = BM25Okapi(tokenized_docs)

    print("✅ Embeddings and BM25 index successfully created!")

    # Save BM25 index
    with open(f"{persist_directory}/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(f"{persist_directory}/bm25_docs.pkl", "wb") as f:
        pickle.dump(chroma_documents, f)

if __name__ == "__main__":
    main()
