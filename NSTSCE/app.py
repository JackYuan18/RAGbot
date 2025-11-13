from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import sys
import os
import logging
import shutil
import webbrowser
import threading
import time
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATABASE_DIR = CURRENT_DIR / "Database"

# Ensure the RAGSystem module is importable regardless of project layout.
for candidate in (
    PROJECT_ROOT / "RAGSystem.py",
    PROJECT_ROOT / "RAG system" / "RAGSystem.py",
    CURRENT_DIR / "RAGSystem.py",
):
    if candidate.exists():
        module_dir = str(candidate.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

from RAGSystem import RAGSystem, RAGConfig, setup_logging, check_gpu_availability

# Initialize RAG system
rag_system = None
browser_opened = False

def initialize_rag_system():
    """Initialize the RAG system with documents."""
    global rag_system
    try:
        # Setup logging
        setup_logging('INFO')
        
        # Initialize RAG system with configuration (device will be auto-detected)
        config = RAGConfig(
            chunk_size=1500,
            retrieval_k=2,
            max_length=1500
            # device=None will trigger auto-detection
        )
        
        rag_system = RAGSystem(config)
        
        # Load documents from Database directory relative to this file
        rag_system.load_documents(str(DATABASE_DIR))
        
        logging.info("RAG system initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize RAG system: {e}")
        return False

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global rag_system
    
    if rag_system is None:
        return jsonify({'response': 'RAG system not initialized. Please try again.', 'sources': []})
    
    user_message = request.json.get('message')
    if user_message:
        try:
            # Process query using RAG system
            response, answer, sources = rag_system.query(user_message)
            
            # Format sources for frontend
            formatted_sources = []
            for source in sources:
                formatted_sources.append({
                    'filename': source.get('filename', 'Unknown Document'),
                    'relevance_score': source.get('relevance_score', 0.0),
                    'rank': source.get('rank', 0),
                    'text_preview': source.get('text_preview', ''),
                    'download_url': f'/api/documents/{source.get("chunk_index", 0)}',
                    'file_url': f'/api/documents/file/{source.get("filename", "unknown")}',
                    'chunk': source.get('chunk', '')
                })
            
            return jsonify({
                'response': answer,
                'sources': formatted_sources
            })
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return jsonify({'response': f'Sorry, there was an error processing your query: {str(e)}', 'sources': []})
    
    return jsonify({'response': "Sorry, no message received.", 'sources': []})

@app.route('/status')
def status():
    """Check RAG system status."""
    global rag_system
    if rag_system is None:
        return jsonify({'status': 'not_initialized', 'message': 'RAG system not initialized'})
    
    try:
        info = rag_system.get_system_info()
        gpu_info = check_gpu_availability()
        info['gpu_info'] = gpu_info
        return jsonify({'status': 'ready', 'info': info})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/gpu/status')
def gpu_status():
    """Check GPU status and availability."""
    try:
        gpu_info = check_gpu_availability()
        return jsonify(gpu_info)
    except Exception as e:
        return jsonify({'error': str(e)})

# File upload endpoint
@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads."""
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'message': 'No files provided'})
        
        files = request.files.getlist('files')
        directory = request.form.get('directory', '')
        
        # Create directory path
        base_dir = DATABASE_DIR
        if directory:
            target_dir = base_dir / directory
        else:
            target_dir = base_dir
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_files = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = target_dir / filename
                file.save(file_path)
                uploaded_files.append(filename)
        
        # Reload RAG system with new documents
        rag_update_info = {'updated': False, 'documents_processed': 0, 'error': None}
        if rag_system:
            try:
                logging.info("Updating RAG system with new documents...")
                rag_system.load_documents(str(base_dir))
                rag_update_info['updated'] = True
                rag_update_info['documents_processed'] = len(rag_system.retriever.documents) if rag_system.retriever.documents else 0
                logging.info(f"RAG system updated successfully with {rag_update_info['documents_processed']} document chunks")
            except Exception as e:
                logging.error(f"Failed to update RAG system: {e}")
                rag_update_info['error'] = str(e)
        
        return jsonify({
            'success': True, 
            'message': f'Uploaded {len(uploaded_files)} files',
            'files': uploaded_files,
            'rag_update': rag_update_info
        })
        
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({'success': False, 'message': str(e)})

# Directory management endpoints
@app.route('/api/directories', methods=['GET'])
def get_directories():
    """Get list of directories."""
    try:
        base_dir = DATABASE_DIR
        directories = {}
        
        if base_dir.exists():
            for item in base_dir.iterdir():
                if item.is_dir():
                    directories[item.name] = {
                        'name': item.name,
                        'path': str(item.relative_to(base_dir)),
                        'created': datetime.fromtimestamp(item.stat().st_ctime).isoformat()
                    }
        
        return jsonify({'directories': directories})
    except Exception as e:
        logging.error(f"Error getting directories: {e}")
        return jsonify({'directories': {}, 'error': str(e)})

@app.route('/api/directories', methods=['POST'])
def create_directory():
    """Create a new directory."""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        parent = data.get('parent', '')
        
        if not name:
            return jsonify({'success': False, 'message': 'Directory name is required'})
        
        base_dir = DATABASE_DIR
        if parent:
            target_dir = base_dir / parent / name
        else:
            target_dir = base_dir / name
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        return jsonify({'success': True, 'message': f'Directory "{name}" created successfully'})
        
    except Exception as e:
        logging.error(f"Error creating directory: {e}")
        return jsonify({'success': False, 'message': str(e)})

# Document management endpoints
@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of documents in a directory."""
    try:
        directory = request.args.get('directory', '')
        base_dir = DATABASE_DIR
        
        if directory:
            target_dir = base_dir / directory
        else:
            target_dir = base_dir
        
        documents = []
        if target_dir.exists():
            for item in target_dir.iterdir():
                if item.is_file():
                    stat = item.stat()
                    documents.append({
                        'name': item.name,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                        'path': str(item.relative_to(base_dir))
                    })
        
        return jsonify({'documents': documents})
    except Exception as e:
        logging.error(f"Error getting documents: {e}")
        return jsonify({'documents': [], 'error': str(e)})

@app.route('/api/documents/<int:chunk_index>', methods=['GET'])
def get_document_content(chunk_index):
    """Get document content by chunk index."""
    global rag_system
    
    if rag_system is None:
        return jsonify({'error': 'RAG system not initialized'}), 404
    
    try:
        if chunk_index >= len(rag_system.retriever.documents):
            return jsonify({'error': 'Document chunk not found'}), 404
        
        document_content = rag_system.retriever.documents[chunk_index]
        metadata = rag_system.retriever.document_metadata[chunk_index] if chunk_index < len(rag_system.retriever.document_metadata) else {}
        
        return jsonify({
            'content': document_content,
            'metadata': metadata
        })
        
    except Exception as e:
        logging.error(f"Error getting document content: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/chunk/<filename>/<int:rank>', methods=['GET'])
def get_document_chunk_by_filename_and_rank(filename, rank):
    """Get document chunk content by filename and retrieval rank."""
    global rag_system
    
    if rag_system is None:
        return jsonify({'error': 'RAG system not initialized'}), 404
    
    try:
        # Find all chunks that match the filename
        matching_chunks = []
        for i, metadata in enumerate(rag_system.retriever.document_metadata):
            if (metadata.get('filename') == filename and 
                i < len(rag_system.retriever.documents)):
                chunk_info = {
                    'index': i,
                    'chunk': rag_system.retriever.documents[i],
                    'metadata': metadata.copy()
                }
                matching_chunks.append(chunk_info)
        
        if not matching_chunks:
            return jsonify({'error': f'No chunks found for filename "{filename}"'}), 404
        
        # If we have multiple chunks, we need to determine which one corresponds to the rank
        # For now, let's return the first chunk and add rank information
        if len(matching_chunks) == 1:
            # Single chunk - return it
            selected_chunk = matching_chunks[0]
        else:
            # Multiple chunks - for now, return the first one
            # In a more sophisticated implementation, we could use the rank to select the right chunk
            selected_chunk = matching_chunks[0]
        
        # Add rank information to metadata
        selected_chunk['metadata']['rank'] = rank
        selected_chunk['metadata']['relevance_score'] = 1.0 / rank  # Simple relevance score based on rank
        
        return jsonify({
            'content': selected_chunk['chunk'],
            'metadata': selected_chunk['metadata']
        })
        
    except Exception as e:
        logging.error(f"Error getting document chunk: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/file/<filename>', methods=['GET'])
def get_document_file(filename):
    """Serve the actual PDF file."""
    global rag_system
    
    if rag_system is None:
        return jsonify({'error': 'RAG system not initialized'}), 404
    
    try:
        # Find the file path from metadata
        file_path = None
        for metadata in rag_system.retriever.document_metadata:
            if metadata.get('filename') == filename:
                file_path = metadata.get('file_path')
                break
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Serve the PDF file
        return send_file(file_path, as_attachment=False, mimetype='application/pdf')
    except Exception as e:
        logging.error(f"Error serving document file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/summary/<filename>', methods=['GET'])
def get_document_summary(filename):
    """Generate and return a summary for a specific document."""
    global rag_system
    
    if rag_system is None:
        return jsonify({'error': 'RAG system not initialized'}), 404
    
    try:
        success, summary = rag_system.generate_document_summary(filename)
        
        if success:
            return jsonify({
                'success': True,
                'summary': summary,
                'filename': filename
            })
        else:
            return jsonify({
                'success': False,
                'error': summary,
                'filename': filename
            }), 400
            
    except Exception as e:
        logging.error(f"Error generating summary for {filename}: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'filename': filename
        }), 500

@app.route('/api/documents/rename', methods=['POST'])
def rename_document():
    """Rename a document."""
    try:
        data = request.get_json()
        old_name = data.get('oldName')
        new_name = data.get('newName')
        directory = data.get('directory', '')
        
        if not old_name or not new_name:
            return jsonify({'success': False, 'message': 'Old name and new name are required'})
        
        base_dir = DATABASE_DIR
        if directory:
            target_dir = base_dir / directory
        else:
            target_dir = base_dir
        
        old_path = target_dir / old_name
        new_path = target_dir / new_name
        
        if not old_path.exists():
            return jsonify({'success': False, 'message': 'File not found'})
        
        if new_path.exists():
            return jsonify({'success': False, 'message': 'File with new name already exists'})
        
        old_path.rename(new_path)
        
        return jsonify({'success': True, 'message': 'File renamed successfully'})
        
    except Exception as e:
        logging.error(f"Error renaming document: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/documents/move', methods=['POST'])
def move_document():
    """Move a document to another directory."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        from_directory = data.get('fromDirectory', '')
        to_directory = data.get('toDirectory', '')
        
        if not filename:
            return jsonify({'success': False, 'message': 'Filename is required'})
        
        base_dir = DATABASE_DIR
        
        # Source path
        if from_directory:
            source_dir = base_dir / from_directory
        else:
            source_dir = base_dir
        
        # Destination path
        if to_directory:
            dest_dir = base_dir / to_directory
        else:
            dest_dir = base_dir
        
        source_path = source_dir / filename
        dest_path = dest_dir / filename
        
        if not source_path.exists():
            return jsonify({'success': False, 'message': 'Source file not found'})
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        if dest_path.exists():
            return jsonify({'success': False, 'message': 'File already exists in destination'})
        
        shutil.move(str(source_path), str(dest_path))
        
        return jsonify({'success': True, 'message': 'File moved successfully'})
        
    except Exception as e:
        logging.error(f"Error moving document: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/documents/delete', methods=['DELETE'])
def delete_document():
    """Delete a document."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        directory = data.get('directory', '')
        
        if not filename:
            return jsonify({'success': False, 'message': 'Filename is required'})
        
        base_dir = DATABASE_DIR
        if directory:
            target_dir = base_dir / directory
        else:
            target_dir = base_dir
        
        file_path = target_dir / filename
        
        if not file_path.exists():
            return jsonify({'success': False, 'message': 'File not found'})
        
        file_path.unlink()
        
        return jsonify({'success': True, 'message': 'File deleted successfully'})
        
    except Exception as e:
        logging.error(f"Error deleting document: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/documents/view', methods=['GET'])
def view_document():
    """View/download a document."""
    try:
        filename = request.args.get('file')
        directory = request.args.get('directory', '')
        
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400
        
        base_dir = Path('Database')
        if directory:
            target_dir = base_dir / directory
        else:
            target_dir = base_dir
        
        file_path = target_dir / filename
        
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(str(file_path), as_attachment=False)
        
    except Exception as e:
        logging.error(f"Error viewing document: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/refresh', methods=['POST'])
def refresh_rag_system():
    """Refresh the RAG system by reloading all documents."""
    global rag_system
    
    try:
        if rag_system is None:
            return jsonify({'success': False, 'message': 'RAG system not initialized'})
        
        base_dir = Path('Database')
        if not base_dir.exists():
            return jsonify({'success': False, 'message': 'Database directory not found'})
        
        logging.info("Manually refreshing RAG system...")
        rag_system.load_documents(str(base_dir))
        
        num_documents = len(rag_system.retriever.documents) if rag_system.retriever.documents else 0
        logging.info(f"RAG system refreshed successfully with {num_documents} document chunks")
        
        return jsonify({
            'success': True,
            'message': f'RAG system refreshed successfully',
            'documents_processed': num_documents
        })
        
    except Exception as e:
        logging.error(f"Error refreshing RAG system: {e}")
        return jsonify({'success': False, 'message': str(e)})

# LLM provider endpoints removed - using only local models


# Authentication endpoints removed - using only local models

@app.route('/api/chatgpt5/toggle', methods=['POST'])
def toggle_chatgpt5():
    """Toggle ChatGPT 5 usage on/off."""
    global rag_system
    
    try:
        data = request.get_json()
        use_chatgpt5 = data.get('use_chatgpt5', False)
        api_key = data.get('api_key', None)
        
        # Reinitialize RAG system with new ChatGPT 5 setting
        if rag_system:
            config = RAGConfig(
                chunk_size=500,
                retrieval_k=2,
                max_length=500,
                use_chatgpt5=use_chatgpt5,
                openai_api_key=api_key
            )
            rag_system = RAGSystem(config)
            
            # Reload documents
            base_dir = Path('Database')
            if base_dir.exists():
                rag_system.load_documents(str(base_dir))
        
        return jsonify({
            'success': True,
            'message': f'ChatGPT 5 {"enabled" if use_chatgpt5 else "disabled"} successfully',
            'use_chatgpt5': use_chatgpt5
        })
        
    except Exception as e:
        logging.error(f"Error toggling ChatGPT 5: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/chatgpt5/validate', methods=['POST'])
def validate_chatgpt5_api_key():
    """Validate ChatGPT 5 API key."""
    try:
        data = request.get_json()
        api_key = data.get('api_key', None)
        
        if not api_key:
            return jsonify({
                'success': False,
                'message': 'API key is required'
            })
        
        # Create a temporary ChatGPT5Automation instance to validate the key
        from ChatGPT5Automation import ChatGPT5Automation
        chatgpt5 = ChatGPT5Automation(api_key=api_key)
        
        # Validate the API key
        is_valid, message = chatgpt5.validate_api_key()
        
        return jsonify({
            'success': is_valid,
            'message': message
        })
        
    except Exception as e:
        logging.error(f"Error validating API key: {e}")
        return jsonify({
            'success': False,
            'message': f'Validation error: {str(e)}'
        })

def open_browser():
    """Open the web browser after a short delay to allow the server to start."""
    time.sleep(1.5)  # Wait for server to start
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    # Clean up browser flag file from previous runs
    browser_flag_file = Path("browser_opened.flag")
    if browser_flag_file.exists():
        browser_flag_file.unlink()
    
    # Initialize RAG system
    if initialize_rag_system():
        print("RAG system initialized successfully")
        
        # Test query
        test_query = "What does human driving data provide?"
        response, answer, sources = rag_system.query(test_query)
        print(f"Test Query: {test_query}")
        print(f"Test Answer: {answer}")
        print(f"Sources: {len(sources)} documents found")
        
        # Start browser opening in a separate thread (only once)
        if not browser_flag_file.exists():
            # Create flag file immediately to prevent multiple threads
            browser_flag_file.touch()
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
        
        print("Opening web interface at http://localhost:5000")
        print("Press CTRL+C to stop the server")
        
        # Start Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize RAG system")
        sys.exit(1)