#!/bin/bash

echo "🚀 Setting up Internship Simulator..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found"

# Navigate to backend directory
cd backend

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env_example.txt .env
    echo "⚠️  Please edit backend/.env and add your API key for AI-powered questions"
    echo "   Recommended: Google Gemini API key (free tier available)"
    echo "   Get your Gemini API key from: https://makersuite.google.com/app/apikey"
    echo "   Alternative: OpenAI API key from: https://platform.openai.com/api-keys"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "1. cd backend"
echo "2. python3 app.py"
echo "3. Open http://localhost:8000 in your browser"
echo ""
echo "Optional: Add your API key to backend/.env for AI-powered questions"
echo "   - Google Gemini (recommended, free tier): https://makersuite.google.com/app/apikey"
echo "   - OpenAI (alternative): https://platform.openai.com/api-keys" 