#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refined RAG System using Object-Oriented Programming
Created on Mon Sep  8 16:13:39 2025

@author: zyuan
"""

import PyPDF2
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import torch as pytorch
import numpy as np
import faiss
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# Import LLM providers
# External LLM providers removed - using only local models
from ChatGPT5Automation import ChatGPT5Automation


class MockGenerator:
    """Mock generator for testing when real models are not available."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, prompt, max_length=500, num_return_sequences=1):
        """Mock call method that returns a simple response."""
        # Extract question from prompt
        if "Question:" in prompt and "Context:" in prompt:
            question = prompt.split("Question:")[1].split("Context:")[0].strip()
        else:
            question = "your question"
        
        # Generate a simple mock response
        mock_response = f"Based on the provided context, here is a response to '{question}': The retrieved documents contain relevant information that can help answer your question. Please note that this is a mock response as the actual language model could not be loaded.\n\n*Generated using Mock Generator (fallback)*"
        
        return [{'generated_text': mock_response}]


class RAGConfig:
    """Configuration class for RAG system parameters."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 retrieval_k: int = 5,
                 max_length: int = 500,
                 model_name: str = 'all-MiniLM-L6-v2',
                 generator_model: str = 'google/flan-t5-large',
                 device: Optional[str] = None,
                 use_chatgpt5: bool = False,
                 openai_api_key: Optional[str] = None):
        self.chunk_size = chunk_size
        self.retrieval_k = retrieval_k
        self.max_length = max_length
        self.model_name = model_name
        self.generator_model = generator_model
        self.use_chatgpt5 = use_chatgpt5
        self.openai_api_key = openai_api_key
        
        # Auto-detect best available device
        if device is None:
            if pytorch.cuda.is_available():
                self.device = 'cuda'
                self.logger = logging.getLogger(__name__)
                self.logger.info(f"GPU detected: {pytorch.cuda.get_device_name(0)}")
                self.logger.info(f"GPU memory: {pytorch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                self.device = 'cpu'
                self.logger = logging.getLogger(__name__)
                self.logger.info("No GPU detected, using CPU")
        else:
            self.device = device
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Using specified device: {self.device}")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'chunk_size': self.chunk_size,
            'retrieval_k': self.retrieval_k,
            'max_length': self.max_length,
            'model_name': self.model_name,
            'generator_model': self.generator_model,
            'device': self.device,
            'use_chatgpt5': self.use_chatgpt5,
            'openai_api_key': self.openai_api_key
        }


class PDFProcessor:
    """Handles PDF text extraction and document chunking."""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
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
            self.logger.error(f"Error reading {pdf_path}: {e}")
            return ""
    
    def load_pdfs_from_directory(self, directory: str) -> Tuple[List[str], List[dict]]:
        """Load text from all PDFs in a directory and chunk them with metadata."""
        documents = []
        metadata = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            self.logger.error(f"Directory {directory} does not exist")
            return documents, metadata
        
        pdf_files = list(directory_path.glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        for pdf_path in pdf_files:
            text = self.extract_text_from_pdf(str(pdf_path))
            if text:
                # Split long text into smaller chunks
                chunks = self._chunk_text(text)
                
                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata.append({
                        'filename': pdf_path.name,
                        'file_path': str(pdf_path),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'text_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk
                    })
                
                self.logger.info(f"Extracted {len(chunks)} chunks from {pdf_path.name}")
        
        return documents, metadata
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of specified size."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
        
        return chunks


class DocumentRetriever:
    """Handles document retrieval using semantic search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = None
        self.index = None
        self.documents = []
        self.document_metadata = []  # Store metadata for each document
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            # For GPU usage, we need to handle device mapping properly
            if self.device == 'cuda' and pytorch.cuda.is_available():
                self.model = SentenceTransformer(self.model_name, device='cuda')
                self.logger.info(f"Initialized retriever model: {self.model_name} on GPU")
            else:
                self.model = SentenceTransformer(self.model_name, device='cpu')
                self.logger.info(f"Initialized retriever model: {self.model_name} on CPU")
        except Exception as e:
            self.logger.error(f"Failed to initialize retriever model: {e}")
            # Fallback to CPU if GPU fails
            if self.device == 'cuda':
                self.logger.warning("GPU initialization failed, falling back to CPU")
                self.device = 'cpu'
                self.model = SentenceTransformer(self.model_name, device='cpu')
                self.logger.info(f"Initialized retriever model: {self.model_name} on CPU (fallback)")
            else:
                raise
    
    def build_index(self, documents: List[str], metadata: List[dict] = None) -> None:
        """Build FAISS index from documents with metadata."""
        if not documents:
            self.logger.warning("No documents provided for indexing")
            return
        
        try:
            # Encode documents
            self.logger.info(f"Encoding {len(documents)} documents...")
            document_embeddings = self.model.encode(documents, convert_to_numpy=True)
            
            # Create FAISS index
            dimension = document_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(document_embeddings)
            
            self.documents = documents
            self.document_metadata = metadata or [{'filename': f'Document {i+1}'} for i in range(len(documents))]
            self.logger.info(f"Built FAISS index with {len(documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to build index: {e}")
            raise
    
    def retrieve_documents(self, query: str, k: int = 5) -> Tuple[List[str], List[dict]]:
        """Retrieve top-k most relevant documents for a query with metadata."""
        if self.index is None or not self.documents:
            self.logger.error("Index not built or no documents available")
            return [], []
        
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            distances, indices = self.index.search(query_embedding, k)
            
            retrieved_docs = []
            retrieved_metadata = []
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    retrieved_docs.append(self.documents[idx])
                    
                    # Get metadata for this document
                    metadata = self.document_metadata[idx].copy() if idx < len(self.document_metadata) else {'filename': f'Document {idx+1}'}
                    metadata['relevance_score'] = float(1.0 / (1.0 + distance))  # Convert distance to similarity score
                    metadata['rank'] = i + 1
                    metadata['chunk'] = self.documents[idx]
                    retrieved_metadata.append(metadata)
                    self.logger.info(f"Retrieved chunk: {self.documents[idx]}")
            
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents for query")
            
            return retrieved_docs, retrieved_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents: {e}")
            return [], []


class AnswerGenerator:
    """Handles answer generation using language models."""
    
    def __init__(self, model_name: str = 'google/flan-t5-large', device: str = 'cpu', use_chatgpt5: bool = False, openai_api_key: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.use_chatgpt5 = use_chatgpt5
        self.openai_api_key = openai_api_key
        self.logger = logging.getLogger(__name__)
        
        # Initialize generator
        self.generator = None
        self.chatgpt5_automation = None
        self._initialize_generator()
    
    def _initialize_generator(self):
        """Initialize the text generation pipeline."""
        try:
            # Always initialize local model as fallback
            self.logger.info("Initializing local model...")
            
            # Determine device ID for transformers pipeline
            if self.device == 'cuda' and pytorch.cuda.is_available():
                device_id = 0  # Use first GPU
                device_name = "GPU"
            else:
                device_id = -1  # Use CPU
                device_name = "CPU"
            
            self.logger.info(f"Using device: {device_name}")
            
            # Try to load the primary model first
            try:
                self.generator = pipeline(
                    "text2text-generation",
                    model=self.model_name,
                    device=device_id,
                    max_length=512
                )
                self.logger.info(f"Initialized generator model: {self.model_name} on {device_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load {self.model_name}, trying fallback model: {e}")
                
                # Fallback to t5-small
                try:
                    self.generator = pipeline(
                        "text2text-generation",
                        model="t5-small",
                        device=device_id,
                        max_length=512
                    )
                    self.logger.info(f"Initialized fallback generator model: t5-small on {device_name}")
                except Exception as e2:
                    self.logger.error(f"Failed to load fallback model: {e2}")
                    # Use mock generator as last resort
                    self.generator = MockGenerator()
                    self.logger.info("Using mock generator as last resort")
            
            # Initialize ChatGPT 5 automation if requested
            if self.use_chatgpt5:
                self.logger.info("Initializing ChatGPT 5 automation...")
                self.chatgpt5_automation = ChatGPT5Automation(api_key=self.openai_api_key)
                self.logger.info("ChatGPT 5 automation initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize generator: {e}")
            # Use mock generator as last resort
            self.generator = MockGenerator()
            self.logger.info("Using mock generator as last resort")
    
    def generate_answer(self, query: str, retrieved_docs: List[str], max_length: int = 500) -> Tuple[Any, str]:
        """Generate answer using retrieved documents as context."""
        if not retrieved_docs:
            self.logger.warning("No retrieved documents provided for answer generation")
            return None, "No relevant documents found to answer your question."
        
        try:
            # Combine retrieved documents into context
            context = " ".join(retrieved_docs)
            
            # Use ChatGPT 5 if enabled
            if self.use_chatgpt5 and self.chatgpt5_automation:
                self.logger.info("Using ChatGPT 5 for answer generation")
                success, answer = self.chatgpt5_automation.generate_response(query, context)
                
                if success:
                    # Add model information to the response
                    model_info = f"\n\n*Generated using ChatGPT 5*"
                    answer_with_model = answer + model_info
                    
                    self.logger.info("Generated answer successfully using ChatGPT 5")
                    return {"generated_text": answer_with_model, "model_used": "chatgpt5"}, answer_with_model
                else:
                    self.logger.error(f"ChatGPT 5 generation failed: {answer}")
                    # Fallback to local model with informative message
                    fallback_message = f"⚠️ ChatGPT 5 automation failed: {answer}\n\n🔄 Falling back to local model...\n\n"
                    self.logger.info("Falling back to local model due to ChatGPT 5 failure")
                    # Continue to local model generation below
            
            # Use local model
            prompt = f"Question: {query}\nContext: {context}\nAnswer:"
            
            # Generate response
            response = self.generator(prompt, max_length=max_length, num_return_sequences=1)
            answer = response[0]['generated_text'].replace("Answer:", "").strip()
            
            # Add fallback message if ChatGPT 5 was attempted
            if self.use_chatgpt5 and 'fallback_message' in locals():
                answer = fallback_message + "\n\n" + answer
            
            # Add model information to the response
            if isinstance(self.generator, MockGenerator):
                model_name = "Mock Generator (fallback)"
            elif hasattr(self.generator, 'model') and 't5' in str(self.generator.model).lower():
                model_name = "Local T5-Small"
            else:
                model_name = "Local Model"
            
            model_info = f"\n\n*Generated using {model_name}*"
            answer_with_model = answer + model_info
            
            self.logger.info("Generated answer successfully using local model")
            return {"generated_text": answer_with_model, "model_used": "local"}, answer_with_model
            
        except Exception as e:
            self.logger.error(f"Failed to generate answer: {e}")
            return None, f"Error generating answer: {str(e)}"


class RAGSystem:
    """Main RAG system that orchestrates all components."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.logger = logging.getLogger(__name__)
        
        # Log device information
        self.logger.info(f"Initializing RAG system on device: {self.config.device}")
        if self.config.device == 'cuda' and pytorch.cuda.is_available():
            self.logger.info(f"GPU: {pytorch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {pytorch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Initialize components
        self.pdf_processor = PDFProcessor(chunk_size=self.config.chunk_size)
        self.retriever = DocumentRetriever(
            model_name=self.config.model_name,
            device=self.config.device
        )
        self.generator = AnswerGenerator(
            model_name=self.config.generator_model,
            device=self.config.device,
            use_chatgpt5=self.config.use_chatgpt5,
            openai_api_key=self.config.openai_api_key
        )
        
        self.logger.info("RAG system initialized successfully")
    
    def load_documents(self, pdf_directory: str) -> None:
        """Load and index documents from PDF directory."""
        try:
            self.logger.info(f"Loading documents from {pdf_directory}")
            documents, metadata = self.pdf_processor.load_pdfs_from_directory(pdf_directory)
            
            if not documents:
                self.logger.warning("No documents loaded")
                return
            
            self.retriever.build_index(documents, metadata)
            self.logger.info(f"Successfully loaded and indexed {len(documents)} document chunks")
            
        except Exception as e:
            self.logger.error(f"Failed to load documents: {e}")
            raise
    
    def query(self, question: str, k: Optional[int] = None) -> Tuple[Any, str, List[dict]]:
        """Process a query and return the answer with sources."""
        if k is None:
            k = self.config.retrieval_k
        
        try:
            self.logger.info(f"Processing query: {question}")
            
            # Step 1: Retrieve relevant documents with metadata
            retrieved_docs, retrieved_metadata = self.retriever.retrieve_documents(question, k)
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Step 2: Generate answer
            response, answer = self.generator.generate_answer(
                question, 
                retrieved_docs, 
                max_length=self.config.max_length
            )
            
            self.logger.info("Query processed successfully")
            return response, answer, retrieved_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to process query: {e}")
            return None, f"Error processing query: {str(e)}", []
    
    def generate_document_summary(self, filename: str) -> Tuple[bool, str]:
        """Generate a summary for a specific document using all its chunks."""
        try:
            self.logger.info(f"Generating summary for document: {filename}")
            
            # Find all chunks belonging to this document
            document_chunks = []
            for i, metadata in enumerate(self.retriever.document_metadata):
                if metadata.get('filename') == filename:
                    if i < len(self.retriever.documents):
                        document_chunks.append(self.retriever.documents[i])
            
            if not document_chunks:
                self.logger.warning(f"No chunks found for document: {filename}")
                return False, f"No content found for document: {filename}"
            
            # Use the RAG system to generate a summary
            summary_query = "summarize the document"
            context = " ".join(document_chunks)
            
            # Generate summary using the same method as regular queries
            response, summary = self.generator.generate_answer(
                summary_query,
                document_chunks,
                max_length=self.config.max_length
            )
            
            self.logger.info(f"Successfully generated summary for {filename}")
            return True, summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary for {filename}: {e}")
            return False, f"Error generating summary: {str(e)}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics."""
        return {
            'config': self.config.to_dict(),
            'num_documents': len(self.retriever.documents) if self.retriever.documents else 0,
            'index_built': self.retriever.index is not None,
            'model_loaded': self.generator.generator is not None
        }


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and return information."""
    gpu_info = {
        'available': False,
        'device_name': None,
        'memory_gb': None,
        'device_count': 0
    }
    
    if pytorch.cuda.is_available():
        gpu_info['available'] = True
        gpu_info['device_count'] = pytorch.cuda.device_count()
        gpu_info['device_name'] = pytorch.cuda.get_device_name(0)
        gpu_info['memory_gb'] = pytorch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return gpu_info

def setup_logging(level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main function for testing the RAG system."""
    # Setup logging
    setup_logging('INFO')
    logger = logging.getLogger(__name__)
    
    try:
        # Check GPU availability
        gpu_info = check_gpu_availability()
        if gpu_info['available']:
            logger.info(f"GPU Available: {gpu_info['device_name']}")
            logger.info(f"GPU Memory: {gpu_info['memory_gb']:.1f} GB")
            logger.info(f"GPU Count: {gpu_info['device_count']}")
        else:
            logger.info("No GPU available, using CPU")
        
        # Initialize RAG system
        config = RAGConfig(
            chunk_size=1500,
            retrieval_k=2,
            max_length=1500
        )
        
        rag_system = RAGSystem(config)
        
        # Load documents
        cwd = os.getcwd()
        pdf_directory = os.path.join(cwd, "Database")
        rag_system.load_documents(pdf_directory)
        
        # Test query
        query = "What does human driving data provide?"
        response, answer = rag_system.query(query)
        
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        
        # Print system info
        info = rag_system.get_system_info()
        print(f"\nSystem Info: {info}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()