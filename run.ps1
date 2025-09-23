# TTS API Development Runner Script for Windows
# This script helps you run the TTS API in development mode on Windows

param(
    [switch]$Build,
    [switch]$NoCache,
    [switch]$Stop,
    [switch]$Logs,
    [switch]$Help
)

if ($Help) {
    Write-Host "🚀 TTS API Development Environment Setup for Windows" -ForegroundColor Green
    Write-Host "=================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\run.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Cyan
    Write-Host "  -Build     : Force rebuild containers" -ForegroundColor White
    Write-Host "  -NoCache   : Build without cache" -ForegroundColor White
    Write-Host "  -Stop      : Stop all services" -ForegroundColor White
    Write-Host "  -Logs      : Show logs" -ForegroundColor White
    Write-Host "  -Help      : Show this help" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\run.ps1              # Start services" -ForegroundColor White
    Write-Host "  .\run.ps1 -Build       # Rebuild and start" -ForegroundColor White
    Write-Host "  .\run.ps1 -Stop        # Stop services" -ForegroundColor White
    Write-Host "  .\run.ps1 -Logs        # Show logs" -ForegroundColor White
    exit 0
}

function Test-Command {
    param([string]$Command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try {
        Get-Command $Command | Out-Null
        return $true
    }
    catch {
        return $false
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
}

# Check if Docker is installed
if (-not (Test-Command "docker")) {
    Write-Host "❌ Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    Write-Host "   Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Check if Docker Compose is installed
if (-not (Test-Command "docker-compose")) {
    Write-Host "❌ Docker Compose is not installed. Please install Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host "📁 Creating necessary directories..." -ForegroundColor Blue
if (-not (Test-Path "uploads\audio")) {
    New-Item -ItemType Directory -Path "uploads\audio" | Out-Null
}
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "📝 Creating .env file from template..." -ForegroundColor Blue
    Copy-Item ".env.example" ".env"
    Write-Host "⚠️  Please edit .env file and set your GEMINI_API_KEY before running!" -ForegroundColor Yellow
}

# Check if GEMINI_API_KEY is set
$envContent = Get-Content ".env" -Raw
if ($envContent -match "your-gemini-api-key-here") {
    Write-Host "⚠️  WARNING: GEMINI_API_KEY is not set in .env file!" -ForegroundColor Yellow
    Write-Host "   Please edit .env file and set your actual Gemini API key." -ForegroundColor Yellow
    Write-Host "   You can get one from: https://makersuite.google.com/app/apikey" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Do you want to continue anyway? (y/N)"
    if ($continue -notmatch "^[Yy]$") {
        exit 1
    }
}

if ($Stop) {
    Write-Host "🛑 Stopping services..." -ForegroundColor Blue
    docker-compose down
    Write-Host "✅ Services stopped successfully!" -ForegroundColor Green
    exit 0
}

if ($Logs) {
    Write-Host "📋 Showing logs..." -ForegroundColor Blue
    docker-compose logs -f
    exit 0
}

# Build and start services
Write-Host "🐳 Building and starting Docker containers..." -ForegroundColor Blue

$buildArgs = @("up")
if ($Build -or $NoCache) {
    $buildArgs += "--build"
}
if ($NoCache) {
    $buildArgs += "--no-deps"
}
$buildArgs += "-d"

docker-compose @buildArgs

# Wait for services to be ready
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Blue
Start-Sleep -Seconds 10

# Check if services are running
$services = docker-compose ps --format "table {{.Service}}\t{{.Status}}"
Write-Host ""
Write-Host "📊 Service Status:" -ForegroundColor Cyan
Write-Host $services -ForegroundColor White

if (docker-compose ps -q | ForEach-Object { docker inspect $_ --format "{{.State.Status}}" } | Where-Object { $_ -ne "running" }) {
    Write-Host "❌ Some services failed to start. Check logs with: docker-compose logs" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Services are running successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Available endpoints:" -ForegroundColor Cyan
Write-Host "   - API: http://localhost:5000" -ForegroundColor White
Write-Host "   - Health Check: http://localhost:5000/api/v1/health" -ForegroundColor White
Write-Host "   - Redis Commander: http://localhost:8081" -ForegroundColor White
Write-Host "   - Monitoring Dashboard: http://localhost:8080" -ForegroundColor White
Write-Host ""
Write-Host "📊 Database:" -ForegroundColor Cyan
Write-Host "   - PostgreSQL: localhost:5432" -ForegroundColor White
Write-Host "   - Redis: localhost:6379" -ForegroundColor White
Write-Host ""
Write-Host "🛠️  Useful commands:" -ForegroundColor Cyan
Write-Host "   - View logs: docker-compose logs -f" -ForegroundColor White
Write-Host "   - Stop services: .\run.ps1 -Stop" -ForegroundColor White
Write-Host "   - Restart services: docker-compose restart" -ForegroundColor White
Write-Host "   - Run tests: docker-compose exec app pytest" -ForegroundColor White
Write-Host ""
Write-Host "🎯 To test the API:" -ForegroundColor Cyan
Write-Host "   curl http://localhost:5000/api/v1/health" -ForegroundColor White
Write-Host ""
Write-Host "📝 Note: Make sure to set your GEMINI_API_KEY in .env file for TTS functionality!" -ForegroundColor Yellow