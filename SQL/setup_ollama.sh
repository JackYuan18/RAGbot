#!/bin/bash
# Setup script for Ollama and SQLCoder
# This script helps install Ollama and the SQLCoder model

set -e

echo "=========================================="
echo "Ollama & SQLCoder Setup Script"
echo "=========================================="
echo ""

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    echo "  Version: $OLLAMA_VERSION"
else
    echo "✗ Ollama is not installed"
    echo ""
    echo "Installing Ollama..."
    echo "This will download and install Ollama to your system."
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
        echo ""
        echo "✓ Ollama installed successfully"
    else
        echo "Installation cancelled. You can install Ollama manually:"
        echo "  curl https://ollama.ai/install.sh | sh"
        exit 1
    fi
fi

echo ""
echo "Checking if Ollama is running..."

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama is running"
else
    echo "✗ Ollama is not running"
    echo ""
    echo "Starting Ollama..."
    # Start Ollama in background
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    echo "  Started Ollama (PID: $OLLAMA_PID)"
    
    # Wait for Ollama to start
    echo "  Waiting for Ollama to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "  ✓ Ollama is ready"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  ⚠ Ollama may still be starting. Please wait a moment and check manually."
    fi
fi

echo ""
echo "Checking for SQLCoder model..."

# Check if SQLCoder model exists
MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | grep -i sqlcoder || echo "")

if [ -z "$MODELS" ]; then
    echo "✗ SQLCoder model not found"
    echo ""
    echo "Installing SQLCoder model..."
    echo "This will download ~4GB. It may take a while..."
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ollama pull sqlcoder
        echo ""
        echo "✓ SQLCoder model installed successfully"
    else
        echo "Installation cancelled. You can install the model manually:"
        echo "  ollama pull sqlcoder"
        exit 1
    fi
else
    echo "✓ SQLCoder model is available"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Ollama should now be running and ready to use."
echo "You can start using SQLCoder in the SQL RAG interface."
echo ""
echo "To verify everything is working:"
echo "  curl http://localhost:11434/api/tags"
echo ""

