#!/bin/bash

# Ollama Setup and Configuration Script

set -e

echo "Starting Ollama setup..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

# Start Ollama service in the background if it's not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve > ollama.log 2>&1 &
    sleep 5
else
    echo "Ollama service is already running."
fi

# Pull recommended models
echo "Pulling models (this may take a while)..."
echo "Pulling llama3.2:1b..."
ollama pull llama3.2:1b

echo "Pulling qwen2.5:0.5b..."
ollama pull qwen2.5:0.5b

echo "Ollama setup complete!"
