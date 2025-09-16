#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:13:39 2025

@author: zyuan
"""

import PyPDF2
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch as pytorch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
            return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""
def generate_answer(query, retrieved_docs):
    # Combine retrieved documents into a context
    context = " ".join(retrieved_docs)
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    response = generator(prompt, max_length=500, num_return_sequences=1)
    return response, response[0]['generated_text'].replace("Answer:", "").strip()
def rag_application(query, k=2):
    # Step 1: Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, k)
    print("Retrieved Documents:", len(retrieved_docs))
    for doc in retrieved_docs:
        print(doc)
    print()
    
    # Step 2: Generate answer using retrieved documents
    response, answer = generate_answer(query, retrieved_docs)
    return response, answer
def load_pdfs_from_directory(directory, chunk_size=1000):
    """Load text from all PDFs in a directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                # Split long text into smaller chunks (optional, for better retrieval)
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]  # Split into ~1000 char chunks
                documents.extend(chunks)
    return documents
# Function to retrieve top-k relevant documents
def retrieve_documents(query, k=5):
    query_embedding = retriever_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [documents[idx] for idx in indices[0]]




if __name__=="__main__":
    cwd = os.getcwd()
    pdf_directory = cwd+"/Database"
    
    documents = load_pdfs_from_directory(pdf_directory, chunk_size=500)
    print(len(documents))
    
    # Load a pre-trained sentence transformer model
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if pytorch.cuda.is_available() else 'cpu')
    # Encode the documents
    document_embeddings = retriever_model.encode(documents, convert_to_numpy=True)
    
    # Create a FAISS index for similarity search
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(document_embeddings)  # Add document embeddings to the index
    
    
    
    # Check if GPU is available
    device = 0 if pytorch.cuda.is_available() else -1
    print(pytorch.cuda.is_available())  # Should print: True
    
    # Load a generative model
    # model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     load_in_4bit=True,  # Quantization for GPU
    #     device_map="auto"   # Auto-map to GPU/CPU
    # )
    # generator = pipeline('text-generation', model=model,tokenizer=tokenizer)
    
    generator = pipeline('text2text-generation', model='google/flan-t5-large', device=device)
    query = "What does human driving data provide?"
    response,answer = rag_application(query, k=2)
    print("Query:", query)
    print("Response:", response)
    print("Answer:", answer)
    # Function to generate an answer
