#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQL Executor Module
Handles SQL generation using SQLCoder via Ollama and executes queries against SQLite database
"""

import sqlite3
import logging
import requests
import json
import re
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import tempfile
import os


class SQLCoderClient:
    """Client for interacting with SQLCoder via Ollama API."""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.model_name = "sqlcoder"
        self.logger = logging.getLogger(__name__)
    
    def generate_sql(self, question: str, schema: str) -> Tuple[bool, str]:
        """
        Generate SQL query from natural language question using SQLCoder.
        
        Args:
            question: Natural language question
            schema: Database schema in CREATE TABLE format
            
        Returns:
            Tuple of (success, sql_query or error_message)
        """
        try:
            # Format prompt according to SQLCoder instructions
            prompt = f"""### Instructions:
Your task is to convert a question into a SQL query, given a Postgres database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{question}`.
This query will run on a database whose schema is represented in this string:
{schema}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
```sql
"""
            
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120  # SQL generation can take time
            )
            
            if response.status_code != 200:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return False, error_msg
            
            result = response.json()
            generated_text = result.get("response", "")
            
            # Extract SQL query from the response
            sql_query = self._extract_sql_from_response(generated_text)
            
            if sql_query:
                self.logger.info(f"Generated SQL query: {sql_query[:100]}...")
                return True, sql_query
            else:
                return False, "Could not extract SQL query from response"
                
        except requests.exceptions.ConnectionError:
            error_msg = (
                f"Could not connect to Ollama at {self.ollama_base_url}.\n\n"
                "Please ensure:\n"
                "1. Ollama is installed: curl https://ollama.ai/install.sh | sh\n"
                "2. Ollama is running: ollama serve\n"
                "3. SQLCoder model is installed: ollama pull sqlcoder"
            )
            self.logger.error(error_msg)
            return False, error_msg
        except requests.exceptions.Timeout:
            error_msg = f"Timeout waiting for Ollama response. Ollama may be busy or the model is loading."
            self.logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Error generating SQL: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def _extract_sql_from_response(self, response: str) -> Optional[str]:
        """Extract SQL query from SQLCoder response."""
        # Look for SQL code blocks
        sql_pattern = r'```sql\s*\n(.*?)```'
        matches = re.findall(sql_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return matches[0].strip()
        
        # If no code block, look for SELECT statements
        select_pattern = r'(SELECT\s+.*?;)'
        matches = re.findall(select_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return matches[0].strip()
        
        # Try to find any SQL-like query
        query_pattern = r'(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+.*?;'
        matches = re.findall(query_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return response  # Return full response as fallback
        
        return None
    
    def check_availability(self) -> Tuple[bool, str]:
        """Check if Ollama and SQLCoder model are available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False, f"Ollama not accessible at {self.ollama_base_url}. Status code: {response.status_code}"
            
            # Check if sqlcoder model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if any("sqlcoder" in name.lower() for name in model_names):
                return True, "SQLCoder model is available"
            else:
                install_msg = f"SQLCoder model not found. Available models: {', '.join(model_names[:5]) if model_names else 'none'}. To install: ollama pull sqlcoder"
                return False, install_msg
                
        except requests.exceptions.ConnectionError:
            install_instructions = (
                f"Ollama is not running at {self.ollama_base_url}.\n\n"
                "To fix this:\n"
                "1. Install Ollama: curl https://ollama.ai/install.sh | sh\n"
                "2. Start Ollama: ollama serve (or it may start automatically)\n"
                "3. Install SQLCoder: ollama pull sqlcoder\n"
                "4. Restart this application"
            )
            return False, install_instructions
        except requests.exceptions.Timeout:
            return False, f"Timeout connecting to Ollama at {self.ollama_base_url}. Ollama may be starting up."
        except Exception as e:
            return False, f"Error checking Ollama: {str(e)}"


class SQLExecutor:
    """Executes SQL queries against SQLite database created from SQL dump file."""
    
    def __init__(self, sql_file_path: str):
        self.sql_file_path = Path(sql_file_path)
        self.logger = logging.getLogger(__name__)
        self.db_path = None
        self.conn = None
    
    def load_sql_file_into_db(self) -> Tuple[bool, str]:
        """
        Load SQL dump file into a temporary SQLite database.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Create temporary database file
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            self.db_path = temp_db.name
            temp_db.close()
            
            self.logger.info(f"Creating SQLite database at {self.db_path}")
            
            # Create connection with optimized settings
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            # Optimize for bulk inserts
            self.conn.execute("PRAGMA journal_mode = MEMORY")
            self.conn.execute("PRAGMA synchronous = OFF")
            self.conn.execute("PRAGMA cache_size = 10000")
            self.conn.execute("PRAGMA temp_store = MEMORY")
            
            # Read SQL file
            self.logger.info(f"Loading SQL file: {self.sql_file_path}")
            file_size = self.sql_file_path.stat().st_size
            self.logger.info(f"SQL file size: {file_size / (1024*1024):.2f} MB")
            
            # Read entire file (or process in chunks for very large files)
            with open(self.sql_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sql_content = f.read()
            
            self.logger.info("Splitting SQL statements...")
            statements = self._split_sql_statements(sql_content)
            self.logger.info(f"Found {len(statements)} SQL statements")
            
            executed = 0
            skipped = 0
            errors = []
            
            # Process statements in batches
            batch_size = 1000
            for i, stmt in enumerate(statements):
                if not stmt.strip() or stmt.strip().startswith('--'):
                    continue
                
                try:
                    # Handle MySQL-specific syntax conversion
                    stmt_modified = self._convert_mysql_to_sqlite(stmt)
                    
                    # Skip empty statements after conversion
                    if not stmt_modified.strip():
                        continue
                    
                    # Execute statement
                    self.conn.execute(stmt_modified)
                    executed += 1
                    
                    # Commit in batches for performance
                    if executed % batch_size == 0:
                        self.conn.commit()
                        self.logger.info(f"Executed {executed} statements ({i+1}/{len(statements)})...")
                    
                except sqlite3.Error as e:
                    error_str = str(e)
                    # Only log first few errors to avoid spam
                    if len(errors) < 10:
                        errors.append(f"Statement {i+1}: {error_str[:100]}")
                    skipped += 1
                    # Continue with next statement
                    continue
                except Exception as e:
                    error_str = str(e)
                    if len(errors) < 10:
                        errors.append(f"Statement {i+1}: {error_str[:100]}")
                    skipped += 1
                    continue
            
            # Final commit
            self.conn.commit()
            self.logger.info(f"Loaded {executed} SQL statements into database (skipped {skipped})")
            
            if errors:
                self.logger.warning(f"Encountered {len(errors)} errors during loading. First few: {errors[:3]}")
            
            # Get table count and sample data count
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_names = [t[0] for t in tables]
            self.logger.info(f"Created {len(tables)} tables: {', '.join(table_names[:5])}")
            
            # Check data in tables
            total_rows = 0
            for table_name in table_names[:10]:  # Check first 10 tables
                try:
                    # Properly quote table name
                    quoted_name = f'"{table_name}"'
                    cursor = self.conn.execute(f"SELECT COUNT(*) FROM {quoted_name}")
                    count = cursor.fetchone()[0]
                    total_rows += count
                    if count > 0:
                        self.logger.info(f"Table {table_name}: {count} rows")
                except Exception as e:
                    self.logger.debug(f"Could not count rows in {table_name}: {e}")
                    pass
            
            message = f"Successfully loaded database with {len(tables)} tables"
            if total_rows > 0:
                message += f" ({total_rows} total rows)"
            
            return True, message
            
        except Exception as e:
            error_msg = f"Error loading SQL file into database: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            if self.conn:
                try:
                    self.conn.close()
                except:
                    pass
            if self.db_path and os.path.exists(self.db_path):
                try:
                    os.unlink(self.db_path)
                except:
                    pass
            return False, error_msg
    
    def _split_sql_statements(self, sql_text: str) -> List[str]:
        """Split SQL text into individual statements."""
        statements = []
        current = ""
        in_string = False
        string_char = None
        escape_next = False
        
        i = 0
        while i < len(sql_text):
            char = sql_text[i]
            
            # Handle escaped characters
            if escape_next:
                current += char
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                escape_next = True
                current += char
                i += 1
                continue
            
            # Handle string delimiters
            if char in ("'", '"', '`') and not in_string:
                in_string = True
                string_char = char
                current += char
            elif char == string_char and in_string:
                # Check if it's escaped
                if not escape_next:
                    in_string = False
                    string_char = None
                current += char
            else:
                current += char
            
            # Check for statement terminator
            if char == ';' and not in_string:
                stmt = current.strip()
                if stmt and not stmt.startswith('--'):
                    statements.append(stmt)
                current = ""
            
            i += 1
        
        # Add remaining statement (if any)
        if current.strip():
            stmt = current.strip()
            if not stmt.startswith('--'):
                statements.append(stmt)
        
        return statements
    
    def _convert_mysql_to_sqlite(self, sql_stmt: str) -> str:
        """Convert MySQL-specific syntax to SQLite-compatible syntax."""
        original = sql_stmt
        
        # Remove backticks (SQLite uses double quotes or no quotes)
        sql_stmt = sql_stmt.replace('`', '')
        
        # Remove MySQL-specific table options
        sql_stmt = re.sub(r'ENGINE\s*=\s*\w+[^;]*', '', sql_stmt, flags=re.IGNORECASE)
        sql_stmt = re.sub(r'DEFAULT\s+CHARSET\s*=\s*\w+', '', sql_stmt, flags=re.IGNORECASE)
        sql_stmt = re.sub(r'COLLATE\s*=\s*\w+', '', sql_stmt, flags=re.IGNORECASE)
        sql_stmt = re.sub(r'AUTO_INCREMENT\s*=\s*\d+', '', sql_stmt, flags=re.IGNORECASE)
        
        # Convert AUTO_INCREMENT to AUTOINCREMENT (but keep in column definition)
        sql_stmt = re.sub(r'\bAUTO_INCREMENT\b', 'AUTOINCREMENT', sql_stmt, flags=re.IGNORECASE)
        
        # Convert MySQL data types to SQLite compatible
        sql_stmt = re.sub(r'\bTINYINT\s*\([^)]+\)', 'INTEGER', sql_stmt, flags=re.IGNORECASE)
        sql_stmt = re.sub(r'\bTINYINT\b', 'INTEGER', sql_stmt, flags=re.IGNORECASE)
        sql_stmt = re.sub(r'\bSMALLINT\s*\([^)]+\)', 'INTEGER', sql_stmt, flags=re.IGNORECASE)
        sql_stmt = re.sub(r'\bMEDIUMINT\s*\([^)]+\)', 'INTEGER', sql_stmt, flags=re.IGNORECASE)
        sql_stmt = re.sub(r'\bBIGINT\s*\([^)]+\)', 'INTEGER', sql_stmt, flags=re.IGNORECASE)
        sql_stmt = re.sub(r'\bDOUBLE\s*\([^)]+\)', 'REAL', sql_stmt, flags=re.IGNORECASE)
        sql_stmt = re.sub(r'\bDOUBLE\b', 'REAL', sql_stmt, flags=re.IGNORECASE)
        sql_stmt = re.sub(r'\bFLOAT\s*\([^)]+\)', 'REAL', sql_stmt, flags=re.IGNORECASE)
        
        # Remove UNSIGNED keyword
        sql_stmt = re.sub(r'\bUNSIGNED\b', '', sql_stmt, flags=re.IGNORECASE)
        
        # Handle DEFAULT NULL - SQLite allows NULL by default
        sql_stmt = re.sub(r'DEFAULT\s+NULL', '', sql_stmt, flags=re.IGNORECASE)
        
        # Remove ON UPDATE clauses (not supported in SQLite)
        sql_stmt = re.sub(r'ON\s+UPDATE\s+[^,\s)]+', '', sql_stmt, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        sql_stmt = re.sub(r'\s+', ' ', sql_stmt)
        sql_stmt = sql_stmt.strip()
        
        return sql_stmt
    
    def execute_query(self, sql_query: str) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute SQL query against the database - DISABLED.
        
        This functionality has been disabled. This method always returns an error.
        
        Args:
            sql_query: SQL query to execute (ignored)
            
        Returns:
            Tuple of (False, None, error_message)
        """
        self.logger.warning("SQL execution attempted but functionality is disabled")
        return False, None, "SQL execution functionality has been disabled"
        try:
            if not self.conn:
                return False, None, "Database not loaded. Please load SQL file first."
            
            # Clean and prepare query
            sql_query = sql_query.strip()
            if not sql_query:
                return False, None, "Empty SQL query"
            
            # Remove trailing semicolon if present (SQLite accepts both)
            if sql_query.endswith(';'):
                sql_query = sql_query[:-1].strip()
            
            # Limit result size to prevent memory issues
            sql_lower = sql_query.lower()
            original_query = sql_query
            
            # For SELECT queries, check if LIMIT is present
            if sql_lower.startswith('select') and 'limit' not in sql_lower:
                # Add reasonable limit
                sql_query = sql_query + ' LIMIT 1000'
                self.logger.info("Added LIMIT 1000 to query")
            
            # Log the query for debugging
            query_preview = sql_query[:200] + "..." if len(sql_query) > 200 else sql_query
            self.logger.info(f"Executing query: {query_preview}")
            
            # Execute query
            cursor = self.conn.execute(sql_query)
            
            # Get column names
            if cursor.description:
                columns = [description[0] for description in cursor.description]
            else:
                # For non-SELECT queries (INSERT, UPDATE, DELETE)
                return True, {
                    'columns': [],
                    'rows': [],
                    'row_count': cursor.rowcount if hasattr(cursor, 'rowcount') else 0,
                    'message': 'Query executed successfully (non-SELECT query)'
                }, None
            
            # Fetch results
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            results = []
            for row in rows:
                result_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    # Convert bytes to string if needed
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except:
                            value = str(value)
                    result_dict[col] = value
                results.append(result_dict)
            
            self.logger.info(f"Query executed successfully, returned {len(results)} rows")
            
            return True, {
                'columns': columns,
                'rows': results,
                'row_count': len(results),
                'query': original_query
            }, None
            
        except sqlite3.OperationalError as e:
            error_msg = f"SQL execution error: {str(e)}. Query: {sql_query[:100]}..."
            self.logger.error(error_msg)
            return False, None, f"SQL Error: {str(e)}"
        except sqlite3.Error as e:
            error_msg = f"SQLite error: {str(e)}"
            self.logger.error(error_msg)
            return False, None, f"Database Error: {str(e)}"
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, None, f"Error: {str(e)}"
    
    def get_schema(self) -> str:
        """Get database schema as CREATE TABLE statements."""
        if not self.conn:
            return ""
        
        try:
            schema = ""
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = cursor.fetchall()
            
            if not tables:
                self.logger.warning("No tables found in database")
                return ""
            
            for table in tables:
                table_name = table[0]
                # Use parameterized query to safely get schema
                cursor = self.conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                create_stmt = cursor.fetchone()
                if create_stmt and create_stmt[0]:
                    schema += create_stmt[0] + ";\n\n"
            
            self.logger.info(f"Retrieved schema for {len(tables)} tables")
            return schema
            
        except Exception as e:
            self.logger.error(f"Error getting schema: {str(e)}", exc_info=True)
            return ""
    
    def close(self):
        """Close database connection and cleanup."""
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.db_path and os.path.exists(self.db_path):
            os.unlink(self.db_path)
            self.logger.info(f"Cleaned up temporary database: {self.db_path}")

