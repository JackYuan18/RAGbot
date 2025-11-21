#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQL RAG System - RAG pipeline for querying SQL database schemas
Similar to RAGSystem.py but specialized for SQL files
"""

import os
import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import torch as pytorch
import numpy as np
import faiss

# Import from parent RAG system
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "RAGsystem"))
from RAGSystem import RAGConfig, DocumentRetriever, AnswerGenerator


class SQLProcessor:
    """Handles SQL file parsing and schema extraction."""
    
    def __init__(self, chunk_size: int = 2000):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    def parse_sql_file(self, sql_path: str) -> Dict[str, Any]:
        """Parse SQL file and extract schema information."""
        try:
            self.logger.info(f"Parsing SQL file: {sql_path}")
            
            # For very large files, read in chunks
            file_size = os.path.getsize(sql_path)
            self.logger.info(f"SQL file size: {file_size / (1024*1024):.2f} MB")
            
            # Read file with error handling for encoding issues
            content = ""
            try:
                with open(sql_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Read first 50MB for schema parsing (CREATE TABLE statements are usually at the beginning)
                    content = f.read(50 * 1024 * 1024)
            except Exception as e:
                self.logger.warning(f"Could not read full file, trying with different encoding: {e}")
                try:
                    with open(sql_path, 'r', encoding='latin-1', errors='ignore') as f:
                        content = f.read(50 * 1024 * 1024)
                except Exception as e2:
                    self.logger.error(f"Failed to read SQL file: {e2}")
                    return {}
            
            # Extract CREATE TABLE statements
            table_schemas = self._extract_table_schemas(content)
            
            # Count INSERT statements (for large files, sample the file)
            insert_info = self._count_insert_statements(sql_path)
            
            # Extract other SQL statements
            other_statements = self._extract_other_statements(content)
            
            return {
                'filename': Path(sql_path).name,
                'file_path': sql_path,
                'file_size': file_size,
                'table_schemas': table_schemas,
                'insert_info': insert_info,
                'other_statements': other_statements,
                'total_tables': len(table_schemas)
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing SQL file {sql_path}: {e}")
            return {}
    
    def _extract_table_schemas(self, content: str) -> Dict[str, Any]:
        """Extract CREATE TABLE statements."""
        table_schemas = {}
        
        # Pattern to match CREATE TABLE statements (multiline)
        # Handle both backticks and no quotes
        pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"]?(\w+)[`"]?\s*\((.*?)\)\s*(?:ENGINE|DEFAULT|;|$)'
        
        # Use re.DOTALL to match across newlines
        matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        
        for match in matches:
            table_name = match.group(1)
            table_definition = match.group(2)
            
            # Clean up table definition
            table_definition = table_definition.strip()
            
            # Extract column definitions
            columns = self._parse_columns(table_definition)
            
            # Create full CREATE statement
            full_create_statement = f"CREATE TABLE `{table_name}` (\n{table_definition}\n)"
            
            # Create formatted schema description
            schema_desc = self._format_schema(table_name, columns, full_create_statement)
            
            table_schemas[table_name] = {
                'name': table_name,
                'columns': columns,
                'create_statement': full_create_statement,
                'schema_description': schema_desc,
                'column_count': len(columns)
            }
        
        self.logger.info(f"Extracted {len(table_schemas)} table schemas")
        return table_schemas
    
    def _parse_columns(self, table_def: str) -> List[Dict[str, Any]]:
        """Parse column definitions from table definition."""
        columns = []
        
        # Split by commas, handling nested parentheses (for CHECK constraints, etc.)
        column_parts = []
        current = ""
        depth = 0
        in_quotes = False
        quote_char = None
        
        for char in table_def:
            if char in ('"', "'", '`') and (depth == 0 or not in_quotes):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
            elif char == '(' and not in_quotes:
                depth += 1
            elif char == ')' and not in_quotes:
                depth -= 1
            elif char == ',' and depth == 0 and not in_quotes:
                column_parts.append(current.strip())
                current = ""
                continue
            
            current += char
        
        if current.strip():
            column_parts.append(current.strip())
        
        for col_def in column_parts:
            col_def = col_def.strip()
            if not col_def:
                continue
            
            # Skip constraint definitions
            if col_def.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK', 'INDEX', 'CONSTRAINT', 'KEY')):
                continue
            
            # Extract column name and type
            # Pattern: column_name TYPE(size) [constraints]
            parts = re.split(r'\s+', col_def, 2)
            if len(parts) >= 2:
                col_name = parts[0].strip('`"\'')
                col_type = parts[1].split('(')[0].upper().strip()  # Get base type without size
                
                # Extract constraints
                constraints = []
                if 'NOT NULL' in col_def.upper():
                    constraints.append('NOT NULL')
                if 'PRIMARY KEY' in col_def.upper():
                    constraints.append('PRIMARY KEY')
                if 'AUTO_INCREMENT' in col_def.upper() or 'AUTOINCREMENT' in col_def.upper():
                    constraints.append('AUTO_INCREMENT')
                if 'DEFAULT' in col_def.upper():
                    default_match = re.search(r'DEFAULT\s+([^,\s)]+)', col_def, re.IGNORECASE)
                    if default_match:
                        default_val = default_match.group(1).strip("'\"`")
                        constraints.append(f"DEFAULT {default_val}")
                
                columns.append({
                    'name': col_name,
                    'type': col_type,
                    'definition': col_def,
                    'constraints': constraints
                })
        
        return columns
    
    def _format_schema(self, table_name: str, columns: List[Dict], create_statement: str) -> str:
        """Format table schema into a readable description."""
        desc = f"Table: {table_name}\n"
        desc += f"Number of Columns: {len(columns)}\n\n"
        desc += "Columns:\n"
        
        for i, col in enumerate(columns, 1):
            col_desc = f"  {i}. {col['name']} ({col['type']})"
            if col['constraints']:
                col_desc += f" - {', '.join(col['constraints'])}"
            desc += col_desc + "\n"
        
        desc += f"\nCREATE TABLE Statement:\n{create_statement}\n"
        
        return desc
    
    def _count_insert_statements(self, sql_path: str) -> Dict[str, int]:
        """Count INSERT statements per table (efficiently for large files)."""
        insert_info = {}
        
        try:
            # For very large files, sample or use regex search
            pattern = r'INSERT\s+(?:INTO\s+)?(?:IF\s+NOT\s+EXISTS\s+)?[`"]?(\w+)[`"]?'
            
            # Read file in chunks
            buffer = ""
            chunk_size = 1024 * 1024  # 1MB chunks
            sample_size = 100 * 1024 * 1024  # Sample first 100MB
            
            try:
                with open(sql_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Read sample
                    sample = f.read(sample_size)
                    matches = re.finditer(pattern, sample, re.IGNORECASE)
                    
                    for match in matches:
                        table_name = match.group(1)
                        insert_info[table_name] = insert_info.get(table_name, 0) + 1
                    
                    # If file is larger, estimate
                    file_size = os.path.getsize(sql_path)
                    if file_size > sample_size:
                        # Estimate total based on sample
                        for table_name in insert_info:
                            # Rough estimate: assume linear distribution
                            ratio = file_size / sample_size
                            insert_info[table_name] = int(insert_info[table_name] * ratio)
                    
            except Exception as e:
                self.logger.warning(f"Could not count INSERT statements: {e}")
                # Try with latin-1 encoding
                with open(sql_path, 'r', encoding='latin-1', errors='ignore') as f:
                    sample = f.read(sample_size)
                    matches = re.finditer(pattern, sample, re.IGNORECASE)
                    for match in matches:
                        table_name = match.group(1)
                        insert_info[table_name] = insert_info.get(table_name, 0) + 1
                        
        except Exception as e:
            self.logger.error(f"Error counting INSERT statements: {e}")
        
        return insert_info
    
    def _extract_other_statements(self, content: str) -> List[str]:
        """Extract other SQL statements (ALTER, CREATE INDEX, etc.)."""
        other_statements = []
        
        patterns = [
            (r'ALTER\s+TABLE\s+[^;]+;', 'ALTER TABLE'),
            (r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+[^;]+;', 'CREATE INDEX'),
        ]
        
        for pattern, statement_type in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                other_statements.append(match.group(0))
        
        return other_statements
    
    def load_sql_files(self, sql_directory: str) -> Tuple[List[str], List[dict]]:
        """Load SQL files and create chunks for RAG."""
        documents = []
        metadata = []
        directory_path = Path(sql_directory)
        
        if not directory_path.exists():
            self.logger.error(f"Directory {sql_directory} does not exist")
            return documents, metadata
        
        sql_files = list(directory_path.glob("*.sql"))
        self.logger.info(f"Found {len(sql_files)} SQL files in {sql_directory}")
        
        for sql_file in sql_files:
            parsed = self.parse_sql_file(str(sql_file))
            
            if not parsed or not parsed.get('table_schemas'):
                self.logger.warning(f"No table schemas found in {sql_file.name}")
                continue
            
            # Create chunks from table schemas
            for table_name, table_info in parsed['table_schemas'].items():
                # Schema description chunk
                schema_chunk = table_info['schema_description']
                
                # Add INSERT info if available
                if table_name in parsed.get('insert_info', {}):
                    row_count = parsed['insert_info'][table_name]
                    schema_chunk += f"\n\nNote: This table has approximately {row_count} INSERT statements (rows of data)."
                
                # Chunk if too large
                chunks = self._chunk_text(schema_chunk)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata.append({
                        'filename': parsed['filename'],
                        'table_name': table_name,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_type': 'schema',
                        'text_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                        'file_path': parsed['file_path'],
                        'column_count': table_info.get('column_count', 0)
                    })
            
            # Add summary chunk for the entire database
            summary_chunk = self._create_database_summary(parsed)
            if summary_chunk:
                documents.append(summary_chunk)
                metadata.append({
                    'filename': parsed['filename'],
                    'table_name': 'DATABASE_SUMMARY',
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'chunk_type': 'summary',
                    'text_preview': summary_chunk[:200] + "..." if len(summary_chunk) > 200 else summary_chunk,
                    'file_path': parsed['file_path']
                })
            
            self.logger.info(f"Extracted {len([d for d in metadata if d.get('filename') == parsed['filename']])} chunks from {parsed['filename']}")
        
        return documents, metadata
    
    def _create_database_summary(self, parsed: Dict) -> str:
        """Create a summary of the entire database."""
        summary = f"Database Schema Summary for {parsed['filename']}\n\n"
        summary += f"File Size: {parsed.get('file_size', 0) / (1024*1024):.2f} MB\n"
        summary += f"Total Tables: {parsed['total_tables']}\n\n"
        
        summary += "Tables Overview:\n"
        for table_name, table_info in parsed['table_schemas'].items():
            col_count = len(table_info['columns'])
            row_info = ""
            if table_name in parsed.get('insert_info', {}):
                row_info = f" (~{parsed['insert_info'][table_name]:,} rows)"
            summary += f"  - {table_name}: {col_count} columns{row_info}\n"
        
        summary += "\nTable Details:\n"
        for table_name, table_info in parsed['table_schemas'].items():
            summary += f"\n{table_name}:\n"
            for col in table_info['columns'][:10]:  # First 10 columns
                summary += f"  - {col['name']} ({col['type']})\n"
            if len(table_info['columns']) > 10:
                summary += f"  ... and {len(table_info['columns']) - 10} more columns\n"
        
        return summary
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of specified size."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks


class SQLRAGSystem:
    """Main SQL RAG System that orchestrates all components."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig(chunk_size=2000)
        self.logger = logging.getLogger(__name__)
        
        # Log device information
        self.logger.info(f"Initializing SQL RAG system on device: {self.config.device}")
        if self.config.device == 'cuda' and pytorch.cuda.is_available():
            self.logger.info(f"GPU: {pytorch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {pytorch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Initialize components
        self.sql_processor = SQLProcessor(chunk_size=self.config.chunk_size)
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
        
        self.logger.info("SQL RAG system initialized successfully")
    
    def load_documents(self, sql_directory: str) -> None:
        """Load and index SQL files from directory."""
        try:
            self.logger.info(f"Loading SQL files from {sql_directory}")
            documents, metadata = self.sql_processor.load_sql_files(sql_directory)
            
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
            
            # Step 1: Retrieve relevant schema chunks with metadata
            retrieved_docs, retrieved_metadata = self.retriever.retrieve_documents(question, k)
            self.logger.info(f"Retrieved {len(retrieved_docs)} schema chunks")
            
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
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics."""
        return {
            'config': self.config.to_dict(),
            'num_documents': len(self.retriever.documents) if self.retriever.documents else 0,
            'index_built': self.retriever.index is not None,
            'model_loaded': self.generator.generator is not None
        }


def setup_logging(level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and return information."""
    from RAGSystem import check_gpu_availability as base_check
    return base_check()


if __name__ == "__main__":
    setup_logging('INFO')
    logger = logging.getLogger(__name__)
    
    try:
        # Check GPU availability
        gpu_info = check_gpu_availability()
        if gpu_info['available']:
            logger.info(f"GPU Available: {gpu_info['device_name']}")
        else:
            logger.info("No GPU available, using CPU")
        
        # Initialize SQL RAG system
        config = RAGConfig(
            chunk_size=2000,
            retrieval_k=3,
            max_length=1500
        )
        
        sql_rag = SQLRAGSystem(config)
        
        # Load SQL files
        sql_directory = Path(__file__).parent
        sql_rag.load_documents(str(sql_directory))
        
        # Test query
        query = "What tables are in this database?"
        response, answer, sources = sql_rag.query(query)
        
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print(f"Sources: {len(sources)} chunks found")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

