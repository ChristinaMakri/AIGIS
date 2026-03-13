# Running AIGIS in Docker

This guide explains how to run the AIGIS simulation in Docker for local testing.

## Prerequisites

- Docker installed on your system ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose (optional, usually included with Docker Desktop)

## Quick Start

### Option 1: Using the Quick Script

```bash
./docker-run.sh
```

This will:
1. Build the Docker image
2. Run the simulation
3. Save visualization snapshots to `output/` directory

### Option 2: Using Docker Compose

```bash
# Build and run
docker-compose up --build

# Or just run (if already built)
docker-compose up
```

### Option 3: Manual Docker Commands

```bash
# Build the image
docker build -t aigis:latest .

# Run headless simulation
docker run --rm -v "$(pwd)/output:/app/output" aigis:latest

# Run with real-time web dashboard (expose port 5000 to host)
docker run --rm -p 5000:5000 -v "$(pwd)/output:/app/output" aigis:latest --web
```

After starting with `--web`, open `http://localhost:5000` in your browser to see the live 7-panel Plotly.js dashboard.

## Output

When running in Docker (headless mode), the simulation prints progress to console and exports results. With `--dashboard`:
- Saves `aigis_dashboard.png` (7-panel overview)
- Batch mode saves `aigis_batch_summary.png`

With `--web`:
- Serves real-time dashboard at `http://localhost:5000`
- Full simulation history is replayed to any browser that connects during or after the run

## Configuration

Edit `src/config.py` before building to change:
- Map location (latitude/longitude)
- Agent counts
- Fire spread parameters
- Simulation duration

Example:
```python
MAP_CENTER_LAT = 38.04  # Athens, Greece
MAP_CENTER_LON = 23.80
MAP_RADIUS = 2000
MAX_STEPS = 500
```

## Advanced Usage

### Web Dashboard in Docker

To access the real-time browser dashboard from the host machine:

```bash
docker run --rm -p 5000:5000 aigis:latest --web --web-port 5000
```

Then open `http://localhost:5000` in your browser. The `--web-port` flag must match the `-p` host mapping.

### With X11 Forwarding (Linux only)

For the static PNG dashboard on Linux (if you need headless PNG output with X11):

1. Allow X11 connections:
```bash
xhost +local:docker
```

2. Uncomment the X11 section in `docker-compose.yml`:
```yaml
    network_mode: host
    environment:
      - DISPLAY=${DISPLAY}
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
```

3. Run normally:
```bash
docker-compose up
```

### Development Mode

To edit code without rebuilding:

Uncomment the volume mounts in `docker-compose.yml`:
```yaml
    volumes:
      - ./src:/app/src
      - ./main.py:/app/main.py
```

### Custom Scenarios

```bash
# Override config at runtime
docker run --rm \
  -v "$(pwd)/output:/app/output" \
  -e MAP_CENTER_LAT=40.7128 \
  -e MAP_CENTER_LON=-74.0060 \
  aigis:latest
```

## Troubleshooting

### "Cannot connect to OpenStreetMap"

This is normal if:
- You're in an area with sparse OSM data
- Network is slow/unavailable
- OSM servers are rate-limiting

The simulation will continue with a minimal fallback graph.

### No output files generated

Check:
1. `output/` directory exists and is writable
2. Docker volume mount is correct: `-v "$(pwd)/output:/app/output"`
3. Simulation ran long enough (snapshots saved every 10 steps)

### Out of memory

Reduce grid size in `src/config.py`:
```python
GRID_WIDTH = 100   # Default: 200
GRID_HEIGHT = 100  # Default: 200
```

## Cleanup

```bash
# Remove container
docker-compose down

# Remove image
docker rmi aigis:latest

# Clean output directory
rm -rf output/*
```

## System Requirements

- **RAM**: 2GB minimum, 4GB recommended
- **Disk**: ~500MB for image + dependencies
- **CPU**: Multi-core recommended for faster simulation
- **Network**: Required for fetching OpenStreetMap data

## Notes

- First run will take longer (downloading dependencies)
- Subsequent runs are faster (Docker caching)
- Monte Carlo batch runs use `RANDOM_SEED = None` for stochastic variance across runs
- All computation is local - no cloud services used
