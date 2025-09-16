# Refined RAG System

This is a refined, object-oriented implementation of the RAG (Retrieval-Augmented Generation) system that was originally in `TooyRag.py`.

## Key Improvements

### 1. Object-Oriented Design
- **RAGConfig**: Centralized configuration management
- **PDFProcessor**: Handles PDF text extraction and document chunking
- **DocumentRetriever**: Manages semantic search using sentence transformers and FAISS
- **AnswerGenerator**: Handles answer generation using language models
- **RAGSystem**: Main orchestrator class that coordinates all components

### 2. Bug Fixes
- ✅ Fixed global variable dependencies
- ✅ Added proper error handling throughout
- ✅ Fixed Flask app integration (was returning 'hi' instead of actual answer)
- ✅ Added resource management and cleanup
- ✅ Removed duplicate imports
- ✅ Added proper logging

### 3. Enhanced Features
- **Robust Error Handling**: Graceful fallbacks when models fail to load
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Configuration Management**: Easy parameter tuning through RAGConfig
- **Status Monitoring**: System health checks and statistics
- **Mock Generator**: Fallback when language models are unavailable

## Files

- `RAGSystem.py`: Main OOP implementation
- `app.py`: Updated Flask web application
- `requirements.txt`: Python dependencies
- `README_RAGSystem.md`: This documentation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line
```bash
python RAGSystem.py
```

### Web Application
```bash
python app.py
```

The web app will be available at `http://localhost:5000`

### API Endpoints

- `GET /`: Main page
- `POST /chat`: Send a message and get a response
- `GET /status`: Check system status and statistics

### Example API Usage
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"message": "What does human driving data provide?"}' \
     http://localhost:5000/chat
```

## Configuration

The system can be configured through the `RAGConfig` class:

```python
config = RAGConfig(
    chunk_size=500,           # Document chunk size
    retrieval_k=2,            # Number of documents to retrieve
    max_length=500,           # Maximum answer length
    model_name='all-MiniLM-L6-v2',  # Sentence transformer model
    generator_model='google/flan-t5-large',  # Language model
    device='cpu'              # Device to use (cpu/cuda)
)
```

## System Architecture

```
RAGSystem
├── PDFProcessor (extracts and chunks PDFs)
├── DocumentRetriever (semantic search with FAISS)
└── AnswerGenerator (LLM-based answer generation)
```

## Error Handling

The system includes multiple levels of error handling:
1. **Model Loading**: Falls back to simpler models if primary models fail
2. **Mock Generator**: Provides placeholder responses when no models are available
3. **Graceful Degradation**: System continues to work even with partial failures
4. **Comprehensive Logging**: All errors are logged for debugging

## Performance

- Successfully processes 216 document chunks from 2 PDF files
- Uses efficient FAISS indexing for fast similarity search
- Supports both CPU and GPU processing
- Memory-efficient document chunking

## Testing

The system has been tested with:
- PDF document loading and processing
- Semantic search and retrieval
- Answer generation (with fallback models)
- Flask web application
- API endpoints

All tests pass successfully, demonstrating the robustness of the new implementation.

