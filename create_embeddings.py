#!/usr/bin/env python
# coding: utf-8

import os
import re
import argparse
import pickle
import pdfplumber
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

def parse_args():
    parser = argparse.ArgumentParser(description="Process a PDF and create embeddings using ChromaDB.")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file.")
    parser.add_argument("--persist-dir", type=str, default="./chroma_db", help="Directory to store embeddings.")
    return parser.parse_args()

def load_embedding_model():
    """Loads sentence-transformers/all-mpnet-base-v2 model and falls back to InLegalBERT if unavailable."""
    model_name = embedding_model_name
    try:
        model = SentenceTransformer(model_name)
        print(f"✅ Using embedding model: {model_name}")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}\nSwitching to fallback model: law-ai/InLegalBERT")
        model = SentenceTransformer("law-ai/InLegalBERT")

    return HuggingFaceEmbeddings(model_name=model_name)

def extract_toc_from_pdf(pdf_path):
    """Extracts the Table of Contents (TOC) from a PDF more robustly."""
    toc_map = {}
    toc_detected = False
    pattern = re.compile(r"^(ARTICLE\s+\d+|\d+\.\d+)\s+([\w\s,;&-]+)\s+(\d+)$", re.MULTILINE)

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue  

            lines = text.split("\n")
            
            # Heuristic to detect TOC start
            if not toc_detected and any("ARTICLE" in line.upper() for line in lines[:10]):  
                toc_detected = True  

            if toc_detected:
                for line in lines:
                    line = line.strip()
                    match = pattern.match(line)
                    if match:
                        section_number, title, page_number = match.groups()
                        toc_map[section_number] = {
                            "title": title.strip(),
                            "page": int(page_number)
                        }

                # Stop extracting if a page doesn't contain TOC-like entries
                if not any(pattern.match(line) for line in lines):
                    break  

    return toc_map


def chunk_text(text, chunk_size=200, overlap=50):
    """
    Splits text into chunks of approximately `chunk_size` words with `overlap` words for context retention.
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]  # If the text is short, return as a single chunk
    
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap  # Move forward with overlap

    return chunks


def extract_rules_with_pdfplumber(pdf_path, toc_map, chunk_size=200, overlap=50):
    """
    Extracts structured HOA rules directly from the PDF using pdfplumber.
    Uses adaptive chunking for better retrieval.
    """
    sections = []
    current_section = None
    current_title = None
    section_text = []
    current_page = None

    # Regex for section detection
    section_pattern = re.compile(r"^(ARTICLE\s+\d+|\d+\.\d+)\s+([A-Za-z].*?)$")

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue  # Skip empty pages

            lines = text.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                match = section_pattern.match(line)
                if match:
                    section_number, section_title = match.groups()

                    # Save previous section with chunking
                    if current_section and section_text:
                        full_text = " ".join(section_text).strip()
                        chunked_texts = chunk_text(full_text, chunk_size, overlap)

                        for chunk in chunked_texts:
                            sections.append({
                                "section": current_section,
                                "title": toc_map.get(current_section, {}).get("title", current_title),
                                "content": chunk,
                                "page": current_page
                            })

                        section_text = []

                    # Start a new section
                    current_section = section_number
                    current_title = section_title
                    current_page = page.page_number  # Save the page number

                section_text.append(line)

    # Save the last section with chunking
    if current_section and section_text:
        full_text = " ".join(section_text).strip()
        chunked_texts = chunk_text(full_text, chunk_size, overlap)

        for chunk in chunked_texts:
            sections.append({
                "section": current_section,
                "title": toc_map.get(current_section, {}).get("title", current_title),
                "content": chunk,
                "page": current_page
            })

    return sections

def main():
    args = parse_args()
    pdf_path = args.pdf_path
    persist_directory = args.persist_dir

    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file '{pdf_path}' not found.")
        return

    print("🔍 Extracting TOC dynamically...")
    toc_map = extract_toc_from_pdf(pdf_path)
    print(f"✅ Extracted {len(toc_map)} TOC entries.")

    print("📑 Extracting structured rules using pdfplumber...")
    structured_rules = extract_rules_with_pdfplumber(pdf_path, toc_map, chunk_size=250)

    if not structured_rules:
        print("❌ No structured rules found. Exiting.")
        return

    print(f"✅ Extracted {len(structured_rules)} rules.")

    chroma_documents = [
        Document(
            page_content=rule["content"],
            metadata={
                "section": rule["section"],
                "title": rule["title"],
                "page": rule["page"]
            }
        )
        for rule in structured_rules if rule["content"].strip()
    ]

    if not chroma_documents:
        print("❌ No valid documents to embed. Exiting.")
        return

    print(f"🔍 Creating embeddings using {embedding_model_name}")
    embedding_model = load_embedding_model()

    print(f"💾 Saving embeddings to {persist_directory} ...")
    os.makedirs(persist_directory, exist_ok=True)

    with tqdm(total=len(chroma_documents), desc="Embedding progress", unit="doc") as pbar:
        try:
            vector_store = Chroma.from_documents(chroma_documents, embedding_model, persist_directory=persist_directory)
            pbar.update(len(chroma_documents))
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return

    print("🔍 Indexing for Hybrid Search (BM25 + Embeddings)...")
    tokenized_docs = [doc.page_content.split(" ") for doc in chroma_documents]
    bm25 = BM25Okapi(tokenized_docs)

    print("✅ Embeddings and BM25 index successfully created!")

    with open(f"{persist_directory}/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(f"{persist_directory}/bm25_docs.pkl", "wb") as f:
        pickle.dump(chroma_documents, f)

if __name__ == "__main__":
    main()
