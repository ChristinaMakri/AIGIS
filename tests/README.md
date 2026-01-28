# AIGIS Unit Tests

## Overview

This directory contains unit tests for the AIGIS simulation system.

## Test Files

### test_fuzzy.py
Tests the fuzzy logic risk assessment system used by the Analyst agent.

**Tests:**
- High wind + near fire → Critical risk
- Far fire + low intensity → Low risk
- Medium conditions → Medium risk
- Risk threshold classifications

**Run:**
```bash
python -m pytest tests/test_fuzzy.py -v
# or
python tests/test_fuzzy.py
```

### test_movement.py
Tests agent movement constraints and the Greenshields traffic model.

**Tests:**
- Agents never exceed maximum speed
- Speed decreases with density
- Gridlock at jam density (speed = 0)
- Confused state reduces speed by 50%
- Greenshields formula accuracy

**Run:**
```bash
python -m pytest tests/test_movement.py -v
# or
python tests/test_movement.py
```

## Running All Tests

```bash
# Using pytest (recommended)
python -m pytest tests/ -v

# Using unittest
python -m unittest discover tests/
```

## Test Coverage

To check test coverage:
```bash
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=html
```

## Requirements

Tests require:
- unittest (built-in)
- pytest (optional, for better output)
- scikit-fuzzy (for fuzzy logic tests)
- numpy
