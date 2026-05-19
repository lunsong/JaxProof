#!/bin/bash
# Run the SSA Verified Code Generator web service

set -e

cd "$(dirname "$0")"

# Load API keys if the file exists
if [ -f "keys" ]; then
    echo "Loading API keys from keys file..."
    source keys
fi

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run:"
    echo "  python3 -m venv venv"
    echo "  venv/bin/pip install flask flask-cors openai"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY is not set. LLM features will be unavailable."
    echo "Set it with: export OPENAI_API_KEY=sk-..."
    echo ""
fi

echo "Starting SSA Verified Code Generator on http://localhost:5000"
exec venv/bin/python app.py
