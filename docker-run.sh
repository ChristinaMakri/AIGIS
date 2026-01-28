#!/bin/bash
# Quick script to build and run AIGIS in Docker

set -e

echo "🐳 Building AIGIS Docker image..."
docker build -t aigis:latest .

echo ""
echo "🚀 Running AIGIS simulation..."
docker run --rm -v "$(pwd)/output:/app/output" aigis:latest

echo ""
echo "✅ Simulation complete! Check the output/ directory for results."
