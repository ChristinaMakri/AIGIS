#!/bin/bash
# Run all AIGIS unit tests

echo "════════════════════════════════════════════════════════════════════════"
echo "🧪 Running AIGIS Unit Tests"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Check if pytest is available
if command -v pytest &> /dev/null; then
    echo "Using pytest..."
    python -m pytest tests/ -v --tb=short
else
    echo "pytest not found, using unittest..."
    python -m unittest discover tests/ -v
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "✅ Tests complete!"
echo "════════════════════════════════════════════════════════════════════════"
