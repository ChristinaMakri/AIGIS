# AIGIS Multi-Agent Disaster Management System
# Docker Image for Local Testing

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set matplotlib to use non-interactive backend by default
ENV MPLBACKEND=Agg

# Create output directory for plots
RUN mkdir -p /app/output

# Default command
CMD ["python3", "main.py"]
