#!/bin/bash

# AIGIS Simulation Runner Script

echo "🛡️  AIGIS: Adaptive Intelligence for Geospatial Incident Simulation"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "▶️  Starting simulation..."
echo ""

# Run the simulation
python main.py

# Deactivate virtual environment
deactivate

echo ""
echo "✅ Simulation complete!"
