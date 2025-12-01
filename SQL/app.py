#!/usr/bin/env python3
"""
SQL RAG Web Interface
Flask application for querying SQL database schemas using RAG
Similar to NSTSCE/app.py but specialized for SQL files
"""

import sys
import os
import subprocess
from pathlib import Path

# Check if running in virtual environment and activate if not
def ensure_venv():
    """Ensure the script is running in the virtual environment."""
    CURRENT_DIR = Path(__file__).resolve().parent
    VENV_DIR = CURRENT_DIR / "venv"
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe" if os.name == 'nt' else VENV_DIR / "bin" / "python"
    
    # Check if venv exists
    if not VENV_PYTHON.exists():
        print(f"Warning: Virtual environment not found at {VENV_DIR}")
        print("Please create it first with: python -m venv venv")
        return False
    
    # Check if we're already running in the venv
    # On Windows, check if sys.executable is in the venv directory
    if os.name == 'nt':
        # Windows
        venv_python_path = str(VENV_PYTHON.resolve())
        current_python = sys.executable
        if venv_python_path.lower() != current_python.lower():
            # Not running in venv, restart with venv's Python
            print(f"Not running in virtual environment. Restarting with venv Python...")
            print(f"Current Python: {current_python}")
            print(f"Venv Python: {venv_python_path}")
            try:
                # Restart the script with venv's Python
                subprocess.run([venv_python_path] + sys.argv, check=True)
                sys.exit(0)
            except subprocess.CalledProcessError as e:
                print(f"Error restarting with venv Python: {e}")
                return False
    else:
        # Unix/Linux/Mac
        venv_python_path = str(VENV_PYTHON.resolve())
        current_python = sys.executable
        if venv_python_path != current_python:
            # Not running in venv, restart with venv's Python
            print(f"Not running in virtual environment. Restarting with venv Python...")
            print(f"Current Python: {current_python}")
            print(f"Venv Python: {venv_python_path}")
            try:
                # Restart the script with venv's Python
                os.execv(venv_python_path, [venv_python_path] + sys.argv)
            except Exception as e:
                print(f"Error restarting with venv Python: {e}")
                return False
    
    return True

# Ensure virtual environment is active
if not ensure_venv():
    print("Failed to activate virtual environment. Continuing anyway...")

from flask import Flask, render_template, request, jsonify, send_file
import logging
import webbrowser
import threading
import time
import re

# Set HuggingFace cache location to use shared cache
# This ensures all Python environments use the same cache
if 'HF_HOME' not in os.environ:
    hf_cache = os.path.expanduser('~/.cache/huggingface')
    os.environ['HF_HOME'] = hf_cache
    logging.info(f"Set HF_HOME to: {hf_cache}")
import json
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

# Try to import DefogSQLCoderClient (optional, requires transformers)
try:
    from DefogSQLCoderClient import DefogSQLCoderClient
    DEFOG_SQLCODER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"DefogSQLCoderClient not available: {e}. Install transformers and torch to use Defog SQLCoder.")
    DefogSQLCoderClient = None
    DEFOG_SQLCODER_AVAILABLE = False

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

# Initialize SQLCoder clients and executors
sqlcoder_client = SQLCoderClient()  # Ollama SQLCoder (default)
defog_sqlcoder_client = None  # Defog SQLCoder (lazy loaded)
sql_executors = {}  # Store SQLExecutor instances per SQL file
current_sqlcoder_type = "ollama"  # "ollama" or "defog"

def get_current_sqlcoder_client():
    """Get the currently selected SQLCoder client."""
    global defog_sqlcoder_client, current_sqlcoder_type
    
    if current_sqlcoder_type == "defog":
        if not DEFOG_SQLCODER_AVAILABLE:
            raise ImportError("Defog SQLCoder is not available. Install transformers and torch.")
        if defog_sqlcoder_client is None:
            defog_sqlcoder_client = DefogSQLCoderClient()
        return defog_sqlcoder_client
    else:
        return sqlcoder_client

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

@app.route('/api/schemas/stored', methods=['GET'])
def get_stored_schemas():
    """Get information about stored schemas (internal use)."""
    global sql_rag_system
    
    if sql_rag_system is None:
        return jsonify({
            'success': False,
            'message': 'SQL RAG system not initialized'
        })
    
    try:
        schemas_info = []
        for filename, schema_data in sql_rag_system.extracted_schemas.items():
            schemas_info.append({
                'filename': filename,
                'file_path': str(schema_data['file_path']),
                'total_tables': schema_data['total_tables'],
                'table_names': list(schema_data['table_schemas'].keys()),
                'file_size_mb': schema_data['file_size'] / (1024 * 1024)
            })
        
        return jsonify({
            'success': True,
            'schemas': schemas_info,
            'total_files': len(schemas_info),
            'cache_dir': str(sql_rag_system.schema_cache_dir),
            'cache_files': [f.name for f in sql_rag_system.schema_cache_dir.glob("*_schema.json")] if sql_rag_system.schema_cache_dir.exists() else []
        })
    except Exception as e:
        logging.error(f"Error getting stored schemas: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/schemas/extract/<filename>', methods=['POST'])
def extract_schema_for_file(filename):
    """Extract and store schema for a specific SQL file."""
    global sql_rag_system
    
    if sql_rag_system is None:
        return jsonify({
            'success': False,
            'message': 'SQL RAG system not initialized'
        })
    
    try:
        sql_file_path = CURRENT_DIR / filename
        if not sql_file_path.exists():
            return jsonify({
                'success': False,
                'message': f'SQL file not found: {filename}'
            })
        
        # Parse the SQL file and extract schema
        logging.info(f"Extracting schema for file: {filename}")
        parsed = sql_rag_system.sql_processor.parse_sql_file(str(sql_file_path))
        
        if not parsed or not parsed.get('table_schemas'):
            return jsonify({
                'success': False,
                'message': f'No table schemas found in {filename}. The file may not contain CREATE TABLE statements.'
            })
        
        # Get database name from parsed data
        # If not found, infer from filename (remove .sql extension and common suffixes)
        database_name = parsed.get('database_name')
        if not database_name:
            # Infer from filename: remove .sql extension and common suffixes
            filename_stem = Path(filename).stem  # Remove .sql extension
            database_name = re.sub(r'_(database|db|sql|dump)$', '', filename_stem, flags=re.IGNORECASE)
            database_name = re.sub(r'^(database|db|sql|dump)_', '', database_name, flags=re.IGNORECASE)
            if not database_name:
                database_name = filename_stem
        
        # Store the extracted schema using database name as key
        sql_rag_system.extracted_schemas[database_name] = {
            'filename': parsed['filename'],
            'file_path': parsed['file_path'],
            'file_size': parsed['file_size'],
            'database_name': database_name,
            'table_schemas': parsed['table_schemas'],
            'total_tables': parsed['total_tables'],
            'insert_info': parsed.get('insert_info', {})
        }
        
        # Save to cache (individual file)
        sql_rag_system._save_schema_to_cache(database_name)
        
        return jsonify({
            'success': True,
            'message': f'Schema extracted successfully for {filename} (database: {database_name})',
            'schema_info': {
                'database_name': database_name,
                'total_tables': parsed['total_tables'],
                'table_names': list(parsed['table_schemas'].keys()),
                'filename': filename
            }
        })
    except Exception as e:
        logging.error(f"Error extracting schema for {filename}: {e}")
        return jsonify({
            'success': False,
            'message': f'Error extracting schema: {str(e)}'
        })

@app.route('/api/schemas/combined', methods=['GET'])
def get_combined_schema():
    """Get combined schema string for all stored schemas."""
    global sql_rag_system
    
    if sql_rag_system is None:
        return jsonify({
            'success': False,
            'message': 'SQL RAG system not initialized'
        })
    
    try:
        table_names = request.args.getlist('table_names')  # Optional query parameter
        table_names = table_names if table_names else None
        
        schema = sql_rag_system.get_combined_schema(table_names=table_names)
        
        return jsonify({
            'success': True,
            'schema': schema,
            'table_names': table_names or sql_rag_system.get_all_table_names()
        })
    except Exception as e:
        logging.error(f"Error getting combined schema: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

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

@app.route('/api/schemas', methods=['GET'])
def get_schemas():
    """Get list of extracted schemas from schema_cache folder."""
    try:
        schemas = []
        
        if sql_rag_system:
            # Get schema cache directory
            cache_dir = sql_rag_system.schema_cache_dir
            
            # Ensure cache directory exists
            if not cache_dir.exists():
                logging.warning(f"Schema cache directory does not exist: {cache_dir}")
                return jsonify({'schemas': [], 'error': f'Schema cache directory not found: {cache_dir}'})
            
            # Read all schema files directly from cache directory
            schema_files = list(cache_dir.glob("*_schema.json"))
            
            if not schema_files:
                logging.info(f"No schema files found in {cache_dir}")
                return jsonify({'schemas': []})
            
            for schema_file in schema_files:
                try:
                    with open(schema_file, 'r', encoding='utf-8') as f:
                        schema_data = json.load(f)
                    
                    # Use the filename (without extension) as schema_name
                    schema_name = str(schema_file.stem)  # e.g., "fars_schema" - ensure it's a string
                    database_name = schema_data.get('database_name', schema_name.replace('_schema', ''))
                    
                    # Ensure schema_name is set
                    if not schema_name:
                        schema_name = f"{database_name}_schema"
                    
                    # Final check: ensure schema_name is always set
                    if not schema_name or schema_name == 'None':
                        schema_name = f"{database_name}_schema" if database_name else schema_file.stem
                    
                    logging.info(f"Loading schema: file={schema_file.name}, schema_name={schema_name}, database_name={database_name}")
                    
                    schema_obj = {
                        'database_name': database_name,
                        'schema_name': schema_name,  # Use filename directly: "fars_schema"
                        'filename': schema_data.get('filename', ''),
                        'total_tables': schema_data.get('total_tables', 0),
                        'table_names': list(schema_data.get('table_schemas', {}).keys()),
                        'file_size': schema_data.get('file_size', 0),
                        'file_size_mb': schema_data.get('file_size', 0) / (1024 * 1024)
                    }
                    
                    # Double-check schema_name is in the object
                    if 'schema_name' not in schema_obj or not schema_obj.get('schema_name'):
                        schema_obj['schema_name'] = f"{database_name}_schema" if database_name else str(schema_file.stem)
                    
                    schemas.append(schema_obj)
                except Exception as e:
                    logging.warning(f"Failed to load schema file {schema_file}: {e}")
                    continue
        else:
            logging.warning("SQL RAG system not initialized")
            return jsonify({'schemas': [], 'error': 'SQL RAG system not initialized'})
        
        return jsonify({'schemas': schemas})
    except Exception as e:
        logging.error(f"Error getting schemas: {e}", exc_info=True)
        return jsonify({'schemas': [], 'error': str(e)})

@app.route('/api/sql/files', methods=['GET'])
def get_sql_files():
    """Get list of SQL files with schema extraction status."""
    try:
        sql_files = []
        # Map database names to filenames
        db_name_to_filename = {}
        if sql_rag_system and sql_rag_system.extracted_schemas:
            for db_name, schema_data in sql_rag_system.extracted_schemas.items():
                filename = schema_data.get('filename', '')
                if filename:
                    db_name_to_filename[filename] = db_name
        
        for sql_file in CURRENT_DIR.glob("*.sql"):
            stat = sql_file.stat()
            filename = sql_file.name
            database_name = db_name_to_filename.get(filename, None)
            has_schema = database_name is not None
            
            # Get schema info if available
            schema_info = None
            if has_schema and sql_rag_system:
                schema_data = sql_rag_system.extracted_schemas[database_name]
                schema_info = {
                    'database_name': database_name,
                    'total_tables': schema_data.get('total_tables', 0),
                    'table_names': list(schema_data.get('table_schemas', {}).keys())
                }
            
            sql_files.append({
                'name': filename,
                'size': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                'schema_extracted': has_schema,
                'database_name': database_name,
                'schema_info': schema_info
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
    """Check SQLCoder availability."""
    try:
        client = get_current_sqlcoder_client()
        available, message = client.check_availability()
        # Format message for HTML display (convert newlines to <br>)
        formatted_message = message.replace('\n', '<br>')
        return jsonify({
            'available': available,
            'message': message,
            'formatted_message': formatted_message,
            'type': current_sqlcoder_type
        })
    except Exception as e:
        error_msg = f'Error checking SQLCoder: {str(e)}'
        logging.error(f"Error checking SQLCoder status: {e}", exc_info=True)
        return jsonify({
            'available': False,
            'message': error_msg,
            'formatted_message': error_msg,
            'type': current_sqlcoder_type
        })

@app.route('/api/sqlcoder/type', methods=['GET', 'POST'])
def sqlcoder_type():
    """Get or set the SQLCoder type (ollama or defog)."""
    global current_sqlcoder_type, defog_sqlcoder_client
    
    if request.method == 'GET':
        return jsonify({
            'type': current_sqlcoder_type,
            'available_types': ['ollama', 'defog']
        })
    
    # POST - set SQLCoder type
    try:
        data = request.get_json()
        new_type = data.get('type', 'ollama')
        
        if new_type not in ['ollama', 'defog']:
            return jsonify({
                'success': False,
                'message': f'Invalid SQLCoder type: {new_type}. Must be "ollama" or "defog"'
            }), 400
        
        current_sqlcoder_type = new_type
        
        # Initialize Defog client if switching to it
        if new_type == 'defog':
            if not DEFOG_SQLCODER_AVAILABLE:
                return jsonify({
                    'success': False,
                    'message': 'Defog SQLCoder is not available. Please install required dependencies: pip install transformers torch'
                }), 400
            
            if defog_sqlcoder_client is None:
                try:
                    defog_sqlcoder_client = DefogSQLCoderClient()
                    logging.info("Defog SQLCoder client initialized")
                except Exception as e:
                    logging.error(f"Failed to initialize Defog SQLCoder: {e}", exc_info=True)
                    error_msg = str(e)
                    # If it's a connection/model download error, provide helpful instructions
                    if "Failed to connect" in error_msg or "not found" in error_msg or "couldn't find it" in error_msg:
                        return jsonify({
                            'success': False,
                            'message': error_msg  # Already contains download instructions
                        }), 500
                    else:
                        return jsonify({
                            'success': False,
                            'message': f'Failed to initialize Defog SQLCoder: {error_msg}\n\nMake sure transformers and torch are installed: pip install transformers torch'
                        }), 500
        
        return jsonify({
            'success': True,
            'type': current_sqlcoder_type,
            'message': f'SQLCoder type set to {current_sqlcoder_type}'
        })
    except Exception as e:
        logging.error(f"Error setting SQLCoder type: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error setting SQLCoder type: {str(e)}'
        }), 500

@app.route('/api/sqlcoder/generate', methods=['POST'])
def generate_sql():
    """Generate SQL query from natural language using SQLCoder with stored schemas."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        filename = data.get('filename', '')
        table_names = data.get('table_names', None)  # Optional: specific tables to use
        
        if not question:
            return jsonify({
                'success': False,
                'message': 'Question is required'
            })
        
        # Try to get schema from SQLRAGSystem first (stored schemas)
        schema = None
        schema_source = None
        
        if sql_rag_system and sql_rag_system.extracted_schemas:
            # Use stored schemas from SQLRAGSystem
            if filename:
                # Get schema for specific file
                if filename in sql_rag_system.extracted_schemas:
                    schema = sql_rag_system.get_schema_for_tables(
                        list(sql_rag_system.extracted_schemas[filename]['table_schemas'].keys())
                    )
                    schema_source = f"stored_schema_{filename}"
                else:
                    logging.warning(f"Filename {filename} not found in stored schemas, using all schemas")
                    schema = sql_rag_system.get_combined_schema(table_names=table_names)
                    schema_source = "stored_schema_all"
            else:
                # Use all stored schemas
                schema = sql_rag_system.get_combined_schema(table_names=table_names)
                schema_source = "stored_schema_all"
        
        # Fallback: Get schema from SQLExecutor if stored schemas not available
        if not schema and filename:
            logging.info("Stored schemas not available, falling back to SQLExecutor")
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
            schema = executor.get_schema()
            schema_source = "executor_schema"
        
        if not schema:
            return jsonify({
                'success': False,
                'message': 'Could not retrieve database schema. Please load SQL files first.'
            })
        
        # Generate SQL using SQLCoder
        client = get_current_sqlcoder_client()
        logging.info(f"Generating SQL for question: {question} (using {schema_source}, SQLCoder: {current_sqlcoder_type})")
        success, sql_query = client.generate_sql(question, schema)
        
        if success:
            return jsonify({
                'success': True,
                'sql_query': sql_query,
                'message': 'SQL query generated successfully',
                'schema_source': schema_source
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

# SQL execution endpoint removed - execution functionality disabled
# @app.route('/api/sql/execute', methods=['POST'])
# def execute_sql():
#     """Execute SQL query - DISABLED"""
#     return jsonify({
#         'success': False,
#         'message': 'SQL execution functionality has been disabled'
#     }), 403

@app.route('/api/sqlcoder/generate-and-execute', methods=['POST'])
def generate_and_execute_sql():
    """Generate SQL from natural language (execution disabled)."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        database_name = data.get('database_name', '')  # Changed from filename to database_name
        table_names = data.get('table_names', None)
        
        if not question:
            return jsonify({
                'success': False,
                'message': 'Question is required'
            })
        
        # Try to get schema from SQLRAGSystem first (stored schemas)
        schema = None
        schema_source = None
        
        if sql_rag_system and sql_rag_system.extracted_schemas:
            # Use stored schemas from SQLRAGSystem
            if database_name:
                if database_name in sql_rag_system.extracted_schemas:
                    schema = sql_rag_system.get_schema_for_tables(
                        list(sql_rag_system.extracted_schemas[database_name]['table_schemas'].keys())
                    )
                    schema_source = f"stored_schema_{database_name}"
                else:
                    schema = sql_rag_system.get_combined_schema(table_names=table_names)
                    schema_source = "stored_schema_all"
            else:
                schema = sql_rag_system.get_combined_schema(table_names=table_names)
                schema_source = "stored_schema_all"
        
        # Fallback: Get schema from SQLExecutor if stored schemas not available
        # Try to find filename from database_name
        if not schema and database_name:
            # Find the SQL file that corresponds to this database name
            filename = None
            if sql_rag_system and sql_rag_system.extracted_schemas:
                for db_name, schema_data in sql_rag_system.extracted_schemas.items():
                    if db_name == database_name:
                        filename = schema_data.get('filename', '')
                        break
            
            if filename:
                logging.info("Stored schemas not available, falling back to SQLExecutor")
                sql_file_path = CURRENT_DIR / filename
                if not sql_file_path.exists():
                    return jsonify({
                        'success': False,
                        'message': f'SQL file not found: {filename}'
                    })
                
                # Load database if not already loaded (only for schema extraction)
                if filename not in sql_executors:
                    logging.info(f"Loading SQL file into database for schema extraction: {filename}")
                    executor = SQLExecutor(str(sql_file_path))
                    success, message = executor.load_sql_file_into_db()
                    if not success:
                        return jsonify({
                            'success': False,
                            'message': f'Failed to load SQL file: {message}'
                        })
                    sql_executors[filename] = executor
                
                executor = sql_executors[filename]
                schema = executor.get_schema()
                schema_source = "executor_schema"
        
        if not schema:
            return jsonify({
                'success': False,
                'message': 'Could not retrieve database schema. Please extract schema first.'
            })
        
        # Generate SQL using SQLCoder
        client = get_current_sqlcoder_client()
        logging.info(f"Generating SQL for question: {question} (using {schema_source}, SQLCoder: {current_sqlcoder_type})")
        success, sql_query = client.generate_sql(question, schema)
        
        if not success:
            return jsonify({
                'success': False,
                'message': sql_query  # sql_query contains error message here
            })
        
        # SQL execution functionality has been disabled
        # Always return just the generated SQL without executing
        return jsonify({
            'success': True,
            'sql_query': sql_query,
            'results': None,
            'message': 'SQL query generated successfully.',
            'schema_source': schema_source
        })
            
    except Exception as e:
        error_msg = f'Error generating SQL: {str(e)}'
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

