#!/bin/bash

# TTS API Development Runner Script
# This script helps you run the TTS API in development mode

set -e

echo "🚀 TTS API Development Environment Setup"
echo "========================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Docker is installed
if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads/audio logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and set your GEMINI_API_KEY before running!"
fi

# Check if GEMINI_API_KEY is set
if grep -q "your-gemini-api-key-here" .env; then
    echo "⚠️  WARNING: GEMINI_API_KEY is not set in .env file!"
    echo "   Please edit .env file and set your actual Gemini API key."
    echo "   You can get one from: https://makersuite.google.com/app/apikey"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build and start services
echo "🐳 Building and starting Docker containers..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Services are running successfully!"
    echo ""
    echo "🌐 Available endpoints:"
    echo "   - API: http://localhost:5000"
    echo "   - Health Check: http://localhost:5000/api/v1/health"
    echo "   - Redis Commander: http://localhost:8081"
    echo "   - Monitoring Dashboard: http://localhost:8080"
    echo ""
    echo "📊 Database:"
    echo "   - PostgreSQL: localhost:5432"
    echo "   - Redis: localhost:6379"
    echo ""
    echo "🛠️  Useful commands:"
    echo "   - View logs: docker-compose logs -f"
    echo "   - Stop services: docker-compose down"
    echo "   - Restart services: docker-compose restart"
    echo "   - Run tests: docker-compose exec app pytest"
    echo ""
    echo "🎯 To test the API:"
    echo "   curl http://localhost:5000/api/v1/health"
    echo ""
    echo "📝 Note: Make sure to set your GEMINI_API_KEY in .env file for TTS functionality!"
else
    echo "❌ Failed to start services. Check logs with: docker-compose logs"
    exit 1
fi