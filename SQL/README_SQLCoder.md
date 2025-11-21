# SQLCoder Setup Instructions

## Overview
SQLCoder integration allows you to generate SQL queries from natural language questions using Ollama and the SQLCoder model.

## Quick Setup

### Option 1: Automatic Setup (Recommended)
Run the setup script:
```bash
cd /home/zyuan/RAGbot/SQL
./setup_ollama.sh
```

### Option 2: Manual Setup

#### 1. Install Ollama
```bash
curl https://ollama.ai/install.sh | sh
```

#### 2. Start Ollama
```bash
# Ollama may start automatically, but if not:
ollama serve

# Or run in background:
nohup ollama serve > /tmp/ollama.log 2>&1 &
```

#### 3. Install SQLCoder Model
```bash
ollama pull sqlcoder
```

This will download approximately 4.1GB. The model is about 4.1GB in size.

#### 4. Verify Installation
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check if SQLCoder is available
ollama list | grep sqlcoder
```

## Troubleshooting

### Error: "Could not connect to Ollama at http://localhost:11434"

**Solutions:**
1. Check if Ollama is running:
   ```bash
   pgrep -f ollama
   # If no process found, start Ollama:
   ollama serve
   ```

2. Check if port 11434 is listening:
   ```bash
   netstat -tuln | grep 11434
   # Or:
   ss -tuln | grep 11434
   ```

3. Check if Ollama is installed:
   ```bash
   which ollama
   # If not found, install Ollama (see above)
   ```

### Error: "SQLCoder model not found"

**Solution:**
```bash
ollama pull sqlcoder
```

### Ollama is Slow or Timing Out

- The SQLCoder model is large (4.1GB) and may take time to load on first use
- Subsequent queries should be faster
- Make sure you have enough RAM (recommended: 16GB+)

## Usage

Once setup is complete:

1. Start the SQL RAG application:
   ```bash
   cd /home/zyuan/RAGbot/SQL
   python3 app.py
   ```

2. Open the web interface at `http://localhost:5001`

3. Select "SQLCoder (Generate SQL)" mode in the sidebar

4. Select a SQL file from the dropdown

5. Ask a natural language question (e.g., "Show all accident records from 1975")

6. Click "Generate SQL" or press Enter

7. View the generated SQL query and results!

## API Endpoints

- `GET /api/sqlcoder/status` - Check if SQLCoder is available
- `POST /api/sqlcoder/generate` - Generate SQL from natural language
- `POST /api/sql/execute` - Execute a SQL query
- `POST /api/sqlcoder/generate-and-execute` - Generate and execute in one step

## Model Information

- **Model**: SQLCoder (via Ollama)
- **Size**: ~4.1GB
- **Context Window**: 32K tokens
- **Source**: https://ollama.com/library/sqlcoder

## Requirements

- Ollama installed and running
- SQLCoder model pulled: `ollama pull sqlcoder`
- At least 16GB RAM recommended
- Python dependencies: `requests` (already in requirements.txt)

## Notes

- The SQL file is loaded into a temporary SQLite database for query execution
- Generated SQL queries are compatible with SQLite
- Results are limited to 1000 rows by default to prevent memory issues
- Large SQL files may take time to load into the database initially

