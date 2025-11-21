#!/usr/bin/env python3
"""
SQL RAG Web Interface
Flask application for querying SQL database schemas using RAG
Similar to NSTSCE/app.py but specialized for SQL files
"""

from flask import Flask, render_template, request, jsonify, send_file
import sys
import os
import logging
import webbrowser
import threading
import time
from pathlib import Path
from datetime import datetime

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SQL_DIR = CURRENT_DIR  # SQL files are in the current directory

# Ensure the SQLRAGSystem module is importable
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "RAGsystem"))

from SQLRAGSystem import SQLRAGSystem, RAGConfig, setup_logging, check_gpu_availability
from SQLExecutor import SQLCoderClient, SQLExecutor

# Try to import ChatGPT5Automation if it exists
try:
    from ChatGPT5Automation import ChatGPT5Automation
except ImportError:
    # Copy from parent if not found
    import shutil
    parent_chatgpt = PROJECT_ROOT / "NSTSCE" / "ChatGPT5Automation.py"
    if parent_chatgpt.exists():
        shutil.copy(parent_chatgpt, CURRENT_DIR / "ChatGPT5Automation.py")
        from ChatGPT5Automation import ChatGPT5Automation
    else:
        ChatGPT5Automation = None

# Initialize SQL RAG system
sql_rag_system = None
browser_opened = False

# Initialize SQLCoder client and executors
sqlcoder_client = SQLCoderClient()
sql_executors = {}  # Store SQLExecutor instances per SQL file

def initialize_sql_rag_system():
    """Initialize the SQL RAG system with SQL files."""
    global sql_rag_system
    try:
        # Setup logging
        setup_logging('INFO')
        
        # Initialize SQL RAG system with configuration
        config = RAGConfig(
            chunk_size=2000,
            retrieval_k=3,
            max_length=1500
        )
        
        sql_rag_system = SQLRAGSystem(config)
        
        # Load SQL files from current directory
        sql_rag_system.load_documents(str(CURRENT_DIR))
        
        logging.info("SQL RAG system initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize SQL RAG system: {e}")
        return False

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global sql_rag_system
    
    if sql_rag_system is None:
        return jsonify({'response': 'SQL RAG system not initialized. Please try again.', 'sources': []})
    
    user_message = request.json.get('message')
    if user_message:
        try:
            # Process query using SQL RAG system
            response, answer, sources = sql_rag_system.query(user_message)
            
            # Format sources for frontend
            formatted_sources = []
            for source in sources:
                formatted_sources.append({
                    'filename': source.get('filename', 'Unknown SQL File'),
                    'table_name': source.get('table_name', 'N/A'),
                    'chunk_type': source.get('chunk_type', 'unknown'),
                    'relevance_score': source.get('relevance_score', 0.0),
                    'rank': source.get('rank', 0),
                    'text_preview': source.get('text_preview', ''),
                    'chunk': source.get('chunk', ''),
                    'column_count': source.get('column_count', 0)
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
    """Check SQL RAG system status."""
    global sql_rag_system
    if sql_rag_system is None:
        return jsonify({'status': 'not_initialized', 'message': 'SQL RAG system not initialized'})
    
    try:
        info = sql_rag_system.get_system_info()
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

@app.route('/api/sql/files', methods=['GET'])
def get_sql_files():
    """Get list of SQL files."""
    try:
        sql_files = []
        for sql_file in CURRENT_DIR.glob("*.sql"):
            stat = sql_file.stat()
            sql_files.append({
                'name': sql_file.name,
                'size': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            })
        
        return jsonify({'files': sql_files})
    except Exception as e:
        logging.error(f"Error getting SQL files: {e}")
        return jsonify({'files': [], 'error': str(e)})

@app.route('/api/sql/files/<filename>')
def view_sql_file(filename):
    """View/download a SQL file (returns first 1MB for preview)."""
    try:
        file_path = CURRENT_DIR / filename
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        # For very large files, return a preview (first 1MB)
        file_size = file_path.stat().st_size
        preview_size = min(1024 * 1024, file_size)  # 1MB preview
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                preview = f.read(preview_size)
                if file_size > preview_size:
                    preview += f"\n\n... [File truncated - showing first {preview_size / (1024*1024):.1f} MB of {file_size / (1024*1024):.1f} MB total] ..."
            
            return jsonify({
                'filename': filename,
                'content': preview,
                'total_size': file_size,
                'preview_size': preview_size,
                'is_preview': file_size > preview_size
            })
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 500
            
    except Exception as e:
        logging.error(f"Error viewing SQL file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sql/files/<filename>/download', methods=['GET'])
def download_sql_file(filename):
    """Download a SQL file."""
    try:
        file_path = CURRENT_DIR / filename
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(str(file_path), as_attachment=True, mimetype='text/plain')
    except Exception as e:
        logging.error(f"Error downloading SQL file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/<int:chunk_index>', methods=['GET'])
def get_document_content(chunk_index):
    """Get document content by chunk index."""
    global sql_rag_system
    
    if sql_rag_system is None:
        return jsonify({'error': 'SQL RAG system not initialized'}), 404
    
    try:
        if chunk_index >= len(sql_rag_system.retriever.documents):
            return jsonify({'error': 'Document chunk not found'}), 404
        
        document_content = sql_rag_system.retriever.documents[chunk_index]
        metadata = sql_rag_system.retriever.document_metadata[chunk_index] if chunk_index < len(sql_rag_system.retriever.document_metadata) else {}
        
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
    global sql_rag_system
    
    if sql_rag_system is None:
        return jsonify({'error': 'SQL RAG system not initialized'}), 404
    
    try:
        # Find all chunks that match the filename
        matching_chunks = []
        for i, metadata in enumerate(sql_rag_system.retriever.document_metadata):
            if (metadata.get('filename') == filename and 
                i < len(sql_rag_system.retriever.documents)):
                chunk_info = {
                    'index': i,
                    'chunk': sql_rag_system.retriever.documents[i],
                    'metadata': metadata.copy()
                }
                matching_chunks.append(chunk_info)
        
        if not matching_chunks:
            return jsonify({'error': f'No chunks found for filename "{filename}"'}), 404
        
        # Return the first matching chunk (or could use rank to select)
        selected_chunk = matching_chunks[0] if len(matching_chunks) == 1 else matching_chunks[0]
        selected_chunk['metadata']['rank'] = rank
        selected_chunk['metadata']['relevance_score'] = 1.0 / rank if rank > 0 else 0.0
        
        return jsonify({
            'content': selected_chunk['chunk'],
            'metadata': selected_chunk['metadata']
        })
        
    except Exception as e:
        logging.error(f"Error getting document chunk: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/refresh', methods=['POST'])
def refresh_rag_system():
    """Refresh the SQL RAG system by reloading all SQL files."""
    global sql_rag_system
    
    try:
        if sql_rag_system is None:
            return jsonify({'success': False, 'message': 'SQL RAG system not initialized'})
        
        logging.info("Manually refreshing SQL RAG system...")
        sql_rag_system.load_documents(str(CURRENT_DIR))
        
        num_documents = len(sql_rag_system.retriever.documents) if sql_rag_system.retriever.documents else 0
        logging.info(f"SQL RAG system refreshed successfully with {num_documents} document chunks")
        
        return jsonify({
            'success': True,
            'message': 'SQL RAG system refreshed successfully',
            'documents_processed': num_documents
        })
    except Exception as e:
        logging.error(f"Error refreshing SQL RAG system: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/chatgpt5/toggle', methods=['POST'])
def toggle_chatgpt5():
    """Toggle ChatGPT 5 usage on/off."""
    global sql_rag_system
    
    try:
        data = request.get_json()
        use_chatgpt5 = data.get('use_chatgpt5', False)
        api_key = data.get('api_key', None)
        
        # Reinitialize SQL RAG system with new ChatGPT 5 setting
        if sql_rag_system:
            config = RAGConfig(
                chunk_size=2000,
                retrieval_k=3,
                max_length=1500,
                use_chatgpt5=use_chatgpt5,
                openai_api_key=api_key
            )
            sql_rag_system = SQLRAGSystem(config)
            
            # Reload SQL files
            sql_rag_system.load_documents(str(CURRENT_DIR))
        
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
        
        if ChatGPT5Automation is None:
            return jsonify({
                'success': False,
                'message': 'ChatGPT5Automation module not available'
            })
        
        # Create a temporary ChatGPT5Automation instance to validate the key
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

@app.route('/api/sqlcoder/status', methods=['GET'])
def sqlcoder_status():
    """Check SQLCoder/Ollama availability."""
    try:
        available, message = sqlcoder_client.check_availability()
        # Format message for HTML display (convert newlines to <br>)
        formatted_message = message.replace('\n', '<br>')
        return jsonify({
            'available': available,
            'message': message,
            'formatted_message': formatted_message
        })
    except Exception as e:
        error_msg = f'Error checking SQLCoder: {str(e)}'
        return jsonify({
            'available': False,
            'message': error_msg,
            'formatted_message': error_msg
        })

@app.route('/api/sqlcoder/generate', methods=['POST'])
def generate_sql():
    """Generate SQL query from natural language using SQLCoder."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        filename = data.get('filename', '')
        
        if not question:
            return jsonify({
                'success': False,
                'message': 'Question is required'
            })
        
        if not filename:
            return jsonify({
                'success': False,
                'message': 'SQL filename is required'
            })
        
        # Get or create SQLExecutor for this file
        sql_file_path = CURRENT_DIR / filename
        if not sql_file_path.exists():
            return jsonify({
                'success': False,
                'message': f'SQL file not found: {filename}'
            })
        
        # Load database if not already loaded
        if filename not in sql_executors:
            logging.info(f"Loading SQL file into database: {filename}")
            executor = SQLExecutor(str(sql_file_path))
            success, message = executor.load_sql_file_into_db()
            if not success:
                return jsonify({
                    'success': False,
                    'message': f'Failed to load SQL file: {message}'
                })
            sql_executors[filename] = executor
        
        executor = sql_executors[filename]
        
        # Get database schema
        schema = executor.get_schema()
        if not schema:
            return jsonify({
                'success': False,
                'message': 'Could not retrieve database schema'
            })
        
        # Generate SQL using SQLCoder
        logging.info(f"Generating SQL for question: {question}")
        success, sql_query = sqlcoder_client.generate_sql(question, schema)
        
        if success:
            return jsonify({
                'success': True,
                'sql_query': sql_query,
                'message': 'SQL query generated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': sql_query  # sql_query contains error message here
            })
            
    except Exception as e:
        logging.error(f"Error generating SQL: {e}")
        return jsonify({
            'success': False,
            'message': f'Error generating SQL: {str(e)}'
        })

@app.route('/api/sql/execute', methods=['POST'])
def execute_sql():
    """Execute SQL query against the database and return results."""
    try:
        data = request.get_json()
        sql_query = data.get('sql_query', '')
        filename = data.get('filename', '')
        
        if not sql_query:
            return jsonify({
                'success': False,
                'message': 'SQL query is required'
            })
        
        if not filename or filename not in sql_executors:
            return jsonify({
                'success': False,
                'message': 'SQL file not loaded. Please generate SQL first.'
            })
        
        executor = sql_executors[filename]
        
        # Check if database connection is still valid
        if not executor.conn:
            logging.warning(f"Database connection lost for {filename}, reloading...")
            sql_file_path = CURRENT_DIR / filename
            executor = SQLExecutor(str(sql_file_path))
            success, message = executor.load_sql_file_into_db()
            if not success:
                return jsonify({
                    'success': False,
                    'message': f'Failed to reload database: {message}'
                })
            sql_executors[filename] = executor
        
        # Execute query
        logging.info(f"Executing SQL query: {sql_query[:200]}...")
        success, results, error = executor.execute_query(sql_query)
        
        if success:
            return jsonify({
                'success': True,
                'results': results,
                'message': f'Query executed successfully, returned {results.get("row_count", 0)} rows'
            })
        else:
            error_msg = error or 'Unknown error executing query'
            logging.error(f"SQL execution failed: {error_msg}")
            return jsonify({
                'success': False,
                'message': error_msg,
                'sql_query': sql_query
            })
            
    except Exception as e:
        error_msg = f'Error executing SQL: {str(e)}'
        logging.error(error_msg, exc_info=True)
        return jsonify({
            'success': False,
            'message': error_msg
        })

@app.route('/api/sqlcoder/generate-and-execute', methods=['POST'])
def generate_and_execute_sql():
    """Generate SQL from natural language and execute it in one step."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        filename = data.get('filename', '')
        
        if not question or not filename:
            return jsonify({
                'success': False,
                'message': 'Question and filename are required'
            })
        
        # Get or create SQLExecutor for this file
        sql_file_path = CURRENT_DIR / filename
        if not sql_file_path.exists():
            return jsonify({
                'success': False,
                'message': f'SQL file not found: {filename}'
            })
        
        # Load database if not already loaded
        if filename not in sql_executors:
            logging.info(f"Loading SQL file into database: {filename}")
            executor = SQLExecutor(str(sql_file_path))
            success, message = executor.load_sql_file_into_db()
            if not success:
                return jsonify({
                    'success': False,
                    'message': f'Failed to load SQL file: {message}'
                })
            sql_executors[filename] = executor
        
        executor = sql_executors[filename]
        
        # Get database schema
        schema = executor.get_schema()
        if not schema:
            return jsonify({
                'success': False,
                'message': 'Could not retrieve database schema'
            })
        
        # Generate SQL using SQLCoder
        logging.info(f"Generating SQL for question: {question}")
        success, sql_query = sqlcoder_client.generate_sql(question, schema)
        
        if not success:
            return jsonify({
                'success': False,
                'message': sql_query  # sql_query contains error message here
            })
        
        # For generate-and-execute, check if execute flag is set
        should_execute = data.get('execute', False)
        
        if should_execute:
            # Execute the generated SQL
            logging.info(f"Executing generated SQL: {sql_query[:200]}...")
            exec_success, results, error = executor.execute_query(sql_query)
            
            if exec_success:
                return jsonify({
                    'success': True,
                    'sql_query': sql_query,
                    'results': results,
                    'message': f'Query executed successfully, returned {results.get("row_count", 0)} rows'
                })
            else:
                # SQL was generated but execution failed
                error_msg = error or 'Unknown error executing query'
                logging.error(f"SQL execution failed: {error_msg}")
                return jsonify({
                    'success': False,
                    'sql_query': sql_query,
                    'message': f'SQL generated but execution failed: {error_msg}',
                    'results': None
                })
        else:
            # Just return the generated SQL without executing
            return jsonify({
                'success': True,
                'sql_query': sql_query,
                'results': None,
                'message': 'SQL query generated successfully. Click "Execute SQL" to run it.'
            })
            
    except Exception as e:
        error_msg = f'Error in generate-and-execute: {str(e)}'
        logging.error(error_msg, exc_info=True)
        return jsonify({
            'success': False,
            'message': error_msg
        })

def open_browser():
    """Open the web browser after a short delay."""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5001')

if __name__ == '__main__':
    # Clean up browser flag file from previous runs
    browser_flag_file = CURRENT_DIR / "browser_opened.flag"
    if browser_flag_file.exists():
        browser_flag_file.unlink()
    
    # Initialize SQL RAG system
    if initialize_sql_rag_system():
        print("SQL RAG system initialized successfully")
        
        # Test query
        test_query = "What tables are in this database?"
        response, answer, sources = sql_rag_system.query(test_query)
        print(f"Test Query: {test_query}")
        print(f"Test Answer: {answer}")
        print(f"Sources: {len(sources)} chunks found")
        
        # Start browser opening in a separate thread (only once)
        if not browser_flag_file.exists():
            browser_flag_file.touch()
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
        
        print("Opening web interface at http://localhost:5001")
        print("Press CTRL+C to stop the server")
        
        # Start Flask app on port 5001 (different from NSTSCE's 5000)
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("Failed to initialize SQL RAG system")

