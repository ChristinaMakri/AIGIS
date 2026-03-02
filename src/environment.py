"""
Location-Agnostic Environment Builder with Real SRTM Elevation or Perlin Noise Terrain
Dynamically identifies safe zones using OSM tags and map perimeter
"""
import io
import numpy as np
import osmnx as ox
import networkx as nx
from typing import Tuple, Dict, List, Optional, Set
from pathlib import Path
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# Optional Perlin noise for terrain generation
try:
    from noise import pnoise2
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False
    # Fallback: simple random elevation
    def pnoise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0,
                repeatx=None, repeaty=None, base=0):
        """Fallback random elevation when noise module unavailable"""
        np.random.seed(int(x * 1000 + y * 1000))
        return np.random.uniform(-1, 1)

from .config import (
    PERLIN_SCALE,
    PERLIN_OCTAVES,
    PERLIN_PERSISTENCE,
    PERLIN_LACUNARITY,
    PERLIN_BASE_HEIGHT,
    PERLIN_AMPLITUDE,
    SAFE_ZONE_TAGS,
    USE_PERIMETER_AS_SAFE,
    USE_CORINE,
    CORINE_CLC_URL,
    CORINE_CACHE_FILE,
    CLC_TO_NFFL_MAP,
    DEFAULT_FUEL_MODEL
)


class Environment:
    """Container for the simulation environment"""

    def __init__(self, graph: nx.MultiDiGraph, fuel_grid: np.ndarray,
                 obstacle_grid: np.ndarray, elevation_grid: np.ndarray,
                 bounds: Tuple[float, float, float, float],
                 grid_shape: Tuple[int, int],
                 safe_nodes: Set[int],
                 fuel_type_grid: Optional[np.ndarray] = None,
                 radius: float = 2000.0):
        self.graph = graph
        self.fuel_grid = fuel_grid
        self.obstacle_grid = obstacle_grid
        self.elevation_grid = elevation_grid
        self.bounds = bounds  # (min_lon, min_lat, max_lon, max_lat)
        min_lon, min_lat, max_lon, max_lat = bounds
        self.lat_center = (min_lat + max_lat) / 2.0
        self.lon_center = (min_lon + max_lon) / 2.0
        self.grid_shape = grid_shape
        self.safe_nodes = safe_nodes  # Set of node IDs that are safe zones
        self.radius = radius  # Map radius in meters (used for cell→metres conversion)

        # Population density grid (loaded from real OSM data)
        self.population_density = np.zeros(grid_shape)

        # Fuel type grid (NFFL fuel models 1-10)
        if fuel_type_grid is not None:
            self.fuel_type_grid = fuel_type_grid
        else:
            self.fuel_type_grid = np.full(grid_shape, DEFAULT_FUEL_MODEL, dtype=np.int8)
            # Set no-fuel areas to 0
            self.fuel_type_grid[fuel_grid == 0] = 0

        # Fire state grids
        self.fire_grid = np.zeros_like(fuel_grid, dtype=np.int8)
        # 0 = no fuel, 1 = burning, 2 = burnt out, 3 = fuel
        self.fire_grid[fuel_grid > 0] = 3

        # Temperature grid (0-100°C, for risk assessment)
        self.temperature_grid = np.zeros_like(fuel_grid, dtype=np.float32)

        # Weather parameters (can be loaded from real weather data)
        self.temperature = 25.0  # Ambient temperature (°C)
        self.humidity = 30.0  # Relative humidity (%)

        # Tracking
        self.step_count = 0

        # Calculate number of exits (unique safe zones)
        self.num_exits = len(safe_nodes) if safe_nodes else 1  # At least 1 (perimeter)

    def latlon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon coordinates to grid indices"""
        min_lon, min_lat, max_lon, max_lat = self.bounds
        x = int((lon - min_lon) / (max_lon - min_lon) * self.grid_shape[1])
        y = int((max_lat - lat) / (max_lat - min_lat) * self.grid_shape[0])
        x = np.clip(x, 0, self.grid_shape[1] - 1)
        y = np.clip(y, 0, self.grid_shape[0] - 1)
        return y, x

    def grid_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices to lat/lon coordinates"""
        min_lon, min_lat, max_lon, max_lat = self.bounds
        lon = min_lon + (col / self.grid_shape[1]) * (max_lon - min_lon)
        lat = max_lat - (row / self.grid_shape[0]) * (max_lat - min_lat)
        return lat, lon

    def get_nearest_node(self, lat: float, lon: float) -> int:
        """Find nearest road network node to given coordinates"""
        try:
            return ox.distance.nearest_nodes(self.graph, lon, lat)
        except:
            # Fallback if graph is empty or error
            return list(self.graph.nodes())[0] if len(self.graph.nodes()) > 0 else 0

    def is_safe_node(self, node: int) -> bool:
        """Check if a node is a designated safe zone"""
        return node in self.safe_nodes

    def find_nearest_safe_node(self, from_node: int) -> int:
        """
        Find the nearest safe node from the given node.
        Uses Dijkstra's shortest path to find closest safe zone.
        """
        if not self.safe_nodes:
            # Fallback: return any node at map edge
            return self._get_perimeter_node()

        # If already at safe node, return it
        if from_node in self.safe_nodes:
            return from_node

        # Find shortest path lengths to all safe nodes
        min_distance = float('inf')
        nearest_safe = None

        for safe_node in self.safe_nodes:
            try:
                path_length = nx.shortest_path_length(
                    self.graph, from_node, safe_node, weight='length'
                )
                if path_length < min_distance:
                    min_distance = path_length
                    nearest_safe = safe_node
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        # Fallback if no path found to any safe node
        if nearest_safe is None:
            # Try perimeter node as last resort (edges of map are considered safe)
            try:
                nearest_safe = self._get_perimeter_node()
            except:
                # Ultimate fallback: stay at current location
                nearest_safe = from_node

        return nearest_safe

    def _get_perimeter_node(self) -> int:
        """
        Get a node at the map perimeter (fallback for safe zones).
        Returns a node at the edge of the map, which are considered safe.
        """
        if len(self.graph.nodes) == 0:
            raise ValueError("Graph has no nodes")

        # Find nodes at the perimeter (min/max lat/lon)
        min_lon, min_lat, max_lon, max_lat = self.bounds
        perimeter_nodes = []

        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            lat, lon = data['y'], data['x']

            # Check if node is near perimeter (within 5% of bounds)
            margin = 0.05
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon

            is_perimeter = (
                lat <= (min_lat + margin * lat_range) or
                lat >= (max_lat - margin * lat_range) or
                lon <= (min_lon + margin * lon_range) or
                lon >= (max_lon - margin * lon_range)
            )

            if is_perimeter:
                perimeter_nodes.append(node)

        # Return first perimeter node found, or any node as ultimate fallback
        return perimeter_nodes[0] if perimeter_nodes else list(self.graph.nodes())[0]


class LiveMapBuilder:
    """
    Location-Agnostic Environment Builder.
    Generates terrain using real SRTM elevation data (when available) or Perlin Noise fallback.
    Identifies safe zones dynamically from OSM tags and map perimeter.
    """

    def __init__(self, center_lat: float, center_lon: float,
                 radius: float, grid_size: Tuple[int, int] = (200, 200)):
        """
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius: Radius in meters
            grid_size: (height, width) of the grid
        """
        self.center = (center_lat, center_lon)
        self.radius = radius
        self.grid_size = grid_size

    def build(self) -> Environment:
        """Build the complete environment"""
        print("🌍 Building Location-Agnostic Environment...")

        # Step 1: Fetch road network
        print("  📍 Fetching road network from OSM...")
        graph = self._fetch_road_network()

        # Step 2: Calculate bounds
        bounds = self._calculate_bounds(graph)

        # Step 3: Fetch land use features
        print("  🌲 Fetching land use data...")
        forest_geometries = self._fetch_forests(bounds)
        building_geometries = self._fetch_buildings(bounds)

        # Step 4: Rasterize into grids
        print("  🗺️  Rasterizing features...")
        fuel_grid = self._rasterize_features(forest_geometries, bounds)
        obstacle_grid = self._rasterize_features(building_geometries, bounds)

        # Step 5: Generate elevation (real SRTM data or Perlin Noise fallback)
        print("  ⛰️  Loading terrain elevation...")
        elevation_grid = self._generate_perlin_terrain()

        # Step 6: Identify safe zones
        print("  🛡️  Identifying safe zones...")
        safe_nodes = self._identify_safe_zones(graph, bounds)

        # Step 7: Fetch Corine Land Cover fuel types (or derive from forest raster)
        if USE_CORINE:
            corine = self._fetch_corine_fuel(bounds)
        else:
            corine = None

        if corine is not None:
            fuel_type_grid = corine
        else:
            fuel_type_grid = np.where(fuel_grid > 0, 8, DEFAULT_FUEL_MODEL).astype(np.int8)
        fuel_type_grid[obstacle_grid > 0] = 0

        print(f"✅ Environment built! {len(safe_nodes)} safe zones identified.")

        return Environment(
            graph=graph,
            fuel_grid=fuel_grid,
            obstacle_grid=obstacle_grid,
            elevation_grid=elevation_grid,
            bounds=bounds,
            grid_shape=self.grid_size,
            safe_nodes=safe_nodes,
            fuel_type_grid=fuel_type_grid,
            radius=self.radius
        )

    def _fetch_corine_fuel(self, bounds: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        Fetch Corine Land Cover (CLC 2018) data and convert to NFFL fuel type grid.

        1. Check cache first
        2. GET Copernicus WCS URL
        3. Parse GeoTIFF with rasterio
        4. Resize to grid_size if needed
        5. Vectorised CLC→NFFL mapping
        6. Cache result

        Returns fuel_type_grid (int8) or None on failure.
        """
        try:
            import requests
        except ImportError:
            print("  [Corine] requests not available – skipping CLC fetch")
            return None

        # Check cache
        lat, lon = self.center
        cache_path = Path(
            CORINE_CACHE_FILE.format(lat=f"{lat:.4f}", lon=f"{lon:.4f}", radius=int(self.radius))
        )
        if cache_path.exists():
            try:
                cached = np.load(cache_path)
                print(f"  [Corine] Loaded from cache: {cache_path}")
                return cached['fuel_type_grid'].astype(np.int8)
            except Exception:
                pass  # Corrupt cache – re-fetch

        # Build WCS URL
        minx, miny, maxx, maxy = bounds
        url = CORINE_CLC_URL.format(
            minx=minx, miny=miny, maxx=maxx, maxy=maxy,
            width=self.grid_size[1], height=self.grid_size[0]
        )

        try:
            print("  [Corine] Fetching CLC 2018 data from Copernicus...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"  [Corine] WARNING: Could not fetch CLC data: {e}")
            return None

        # Parse GeoTIFF
        try:
            import rasterio
            from scipy.ndimage import zoom as ndimage_zoom

            with rasterio.open(io.BytesIO(response.content)) as src:
                clc_data = src.read(1).astype(np.int16)

            # Resize to match grid_size if needed
            if clc_data.shape != tuple(self.grid_size):
                scale_y = self.grid_size[0] / clc_data.shape[0]
                scale_x = self.grid_size[1] / clc_data.shape[1]
                clc_data = ndimage_zoom(clc_data, (scale_y, scale_x), order=0).astype(np.int16)

            # Vectorised CLC → NFFL mapping
            fuel_type_grid = np.full(self.grid_size, DEFAULT_FUEL_MODEL, dtype=np.int8)
            for clc_code, nffl in CLC_TO_NFFL_MAP.items():
                fuel_type_grid[clc_data == clc_code] = nffl

            # Cache result
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, fuel_type_grid=fuel_type_grid)
            print(f"  [Corine] CLC data loaded and cached to {cache_path}")

            return fuel_type_grid

        except Exception as e:
            print(f"  [Corine] WARNING: Could not parse CLC GeoTIFF: {e}")
            return None

    def _fetch_road_network(self) -> nx.MultiDiGraph:
        """Fetch road network from OSM"""
        try:
            graph = ox.graph_from_point(
                self.center,
                dist=self.radius,
                network_type='drive',
                simplify=True
            )

            # Add travel speeds and times
            graph = ox.add_edge_speeds(graph)
            graph = ox.add_edge_travel_times(graph)

            return graph
        except Exception as e:
            print(f"  ⚠️  Warning: Could not fetch road network: {e}")
            print("  Creating minimal fallback graph...")
            # Create minimal graph
            G = nx.MultiDiGraph()
            G.add_node(0, x=self.center[1], y=self.center[0])
            G.add_node(1, x=self.center[1] + 0.01, y=self.center[0])
            G.add_edge(0, 1, length=1000)
            return G

    def _calculate_bounds(self, graph: nx.MultiDiGraph) -> Tuple[float, float, float, float]:
        """Calculate geographic bounds from graph"""
        if len(graph.nodes) == 0:
            # Fallback
            lat, lon = self.center
            offset = self.radius / 111320
            return (lon - offset, lat - offset, lon + offset, lat + offset)

        lons = [data['x'] for _, data in graph.nodes(data=True)]
        lats = [data['y'] for _, data in graph.nodes(data=True)]

        return (min(lons), min(lats), max(lons), max(lats))

    def _fetch_forests(self, bounds: Tuple[float, float, float, float]) -> list:
        """Fetch forest geometries from OSM"""
        try:
            center_lat = (bounds[1] + bounds[3]) / 2.0
            center_lon = (bounds[0] + bounds[2]) / 2.0
            dist = max(
                abs(bounds[3] - bounds[1]) * 111320 / 2,
                abs(bounds[2] - bounds[0]) * 111320 / 2,
            )
            tags = {
                'landuse': ['forest', 'wood'],
                'natural': ['wood', 'tree', 'tree_row']
            }
            gdf = ox.features_from_point(
                (center_lat, center_lon),
                tags=tags,
                dist=int(dist)
            )
            geoms = []
            for geom in gdf.geometry:
                if geom is not None and not geom.is_empty:
                    try:
                        b = geom.bounds
                        if all(np.isfinite(v) for v in b):
                            geoms.append(geom)
                    except Exception:
                        pass
            return geoms
        except Exception as e:
            print(f"  ⚠️  Could not fetch forests: {e}")
            return []

    def _fetch_buildings(self, bounds: Tuple[float, float, float, float]) -> list:
        """Fetch building geometries from OSM"""
        try:
            center_lat = (bounds[1] + bounds[3]) / 2.0
            center_lon = (bounds[0] + bounds[2]) / 2.0
            dist = max(
                abs(bounds[3] - bounds[1]) * 111320 / 2,
                abs(bounds[2] - bounds[0]) * 111320 / 2,
            )
            tags = {
                'building': True,
                'landuse': ['residential', 'commercial', 'industrial']
            }
            gdf = ox.features_from_point(
                (center_lat, center_lon),
                tags=tags,
                dist=int(dist)
            )
            geoms = []
            for geom in gdf.geometry:
                if geom is not None and not geom.is_empty:
                    try:
                        b = geom.bounds
                        if all(np.isfinite(v) for v in b):
                            geoms.append(geom)
                    except Exception:
                        pass
            return geoms
        except Exception as e:
            print(f"  ⚠️  Could not fetch buildings: {e}")
            return []

    def _rasterize_features(self, geometries: list,
                           bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Convert vector geometries to raster grid"""
        grid = np.zeros(self.grid_size, dtype=np.float32)

        if not geometries:
            return grid

        min_lon, min_lat, max_lon, max_lat = bounds

        for geom in geometries:
            if geom is None or geom.is_empty:
                continue

            try:
                if not hasattr(geom, 'bounds'):
                    continue

                geom_bounds = geom.bounds

                # Skip geometries with NaN or invalid bounds
                if any(not np.isfinite(v) for v in geom_bounds):
                    continue

                # Convert to grid coordinates
                x_min = int((geom_bounds[0] - min_lon) / (max_lon - min_lon) * self.grid_size[1])
                y_min = int((max_lat - geom_bounds[3]) / (max_lat - min_lat) * self.grid_size[0])
                x_max = int((geom_bounds[2] - min_lon) / (max_lon - min_lon) * self.grid_size[1])
                y_max = int((max_lat - geom_bounds[1]) / (max_lat - min_lat) * self.grid_size[0])

                # Clip to grid bounds
                x_min = max(0, min(x_min, self.grid_size[1] - 1))
                x_max = max(0, min(x_max, self.grid_size[1] - 1))
                y_min = max(0, min(y_min, self.grid_size[0] - 1))
                y_max = max(0, min(y_max, self.grid_size[0] - 1))

                # Rasterize
                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        lon = min_lon + (x / self.grid_size[1]) * (max_lon - min_lon)
                        lat = max_lat - (y / self.grid_size[0]) * (max_lat - min_lat)
                        point = Point(lon, lat)

                        if geom.contains(point) or geom.intersects(point.buffer(0.0001)):
                            grid[y, x] = 1.0
            except Exception:
                continue

        return grid

    def _generate_perlin_terrain(self) -> np.ndarray:
        """
        Generate realistic terrain using real SRTM elevation data or Perlin Noise fallback.
        Returns elevation grid in meters.
        """
        # Try to load real elevation data first
        from pathlib import Path
        elevation_file = Path("data/elevation_data.npz")

        if elevation_file.exists():
            try:
                print("  📊 Loading real SRTM elevation data...")
                elev_data = np.load(elevation_file)

                # Check if location matches (within reasonable tolerance)
                saved_lat = float(elev_data['latitude'])
                saved_lon = float(elev_data['longitude'])
                saved_radius = float(elev_data['radius'])

                lat_diff = abs(saved_lat - self.center[0])
                lon_diff = abs(saved_lon - self.center[1])
                radius_diff = abs(saved_radius - self.radius)

                # Allow 0.1 degree (~11km) and 20% radius difference
                if lat_diff < 0.1 and lon_diff < 0.1 and radius_diff < self.radius * 0.2:
                    elevation_grid = elev_data['elevation_grid']

                    # Resize to match current grid_size if needed
                    if elevation_grid.shape != self.grid_size:
                        from scipy.ndimage import zoom
                        scale_y = self.grid_size[0] / elevation_grid.shape[0]
                        scale_x = self.grid_size[1] / elevation_grid.shape[1]
                        elevation_grid = zoom(elevation_grid, (scale_y, scale_x), order=1)

                    print(f"  ✅ Real elevation data loaded (min: {elevation_grid.min():.1f}m, max: {elevation_grid.max():.1f}m)")
                    return elevation_grid.astype(np.float32)
                else:
                    print(f"  ⚠️  Elevation data location mismatch, generating new terrain...")
            except Exception as e:
                print(f"  ⚠️  Could not load elevation data ({e}), using fallback...")

        # Fallback: Generate Perlin Noise terrain
        print("  🎲 Generating Perlin Noise terrain (fallback)...")
        elevation = np.zeros(self.grid_size, dtype=np.float32)

        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                # Generate Perlin noise value
                noise_val = pnoise2(
                    x / PERLIN_SCALE,
                    y / PERLIN_SCALE,
                    octaves=PERLIN_OCTAVES,
                    persistence=PERLIN_PERSISTENCE,
                    lacunarity=PERLIN_LACUNARITY,
                    repeatx=self.grid_size[1],
                    repeaty=self.grid_size[0],
                    base=0
                )

                # Scale to elevation range
                elevation[y, x] = PERLIN_BASE_HEIGHT + (noise_val * PERLIN_AMPLITUDE)

        # Ensure non-negative
        elevation = np.maximum(elevation, 0)

        return elevation

    def _identify_safe_zones(self, graph: nx.MultiDiGraph,
                            bounds: Tuple[float, float, float, float]) -> Set[int]:
        """
        Identify safe nodes dynamically using OSM tags and perimeter nodes.
        Returns a set of node IDs that are designated safe zones.
        """
        safe_nodes = set()

        if len(graph.nodes) == 0:
            return safe_nodes

        # Method 1: Fetch safe zone features from OSM
        try:
            for tag_key, tag_values in SAFE_ZONE_TAGS.items():
                for tag_value in tag_values:
                    try:
                        gdf = ox.features_from_bbox(
                            bbox=(bounds[3], bounds[1], bounds[2], bounds[0]),
                            tags={tag_key: tag_value}
                        )

                        # For each safe feature, find nearby nodes
                        for geom in gdf.geometry:
                            if geom is None or geom.is_empty:
                                continue

                            # Get centroid
                            centroid = geom.centroid
                            nearest_node = ox.distance.nearest_nodes(
                                graph, centroid.x, centroid.y
                            )
                            safe_nodes.add(nearest_node)

                    except Exception as e:
                        continue
        except Exception as e:
            print(f"  ⚠️  Could not fetch safe zones: {e}")

        # Method 2: Add perimeter nodes (map edges) as safe
        if USE_PERIMETER_AS_SAFE:
            perimeter_nodes = self._get_perimeter_nodes(graph, bounds)
            safe_nodes.update(perimeter_nodes)

        # Fallback: If no safe nodes found, use perimeter
        if not safe_nodes and len(graph.nodes) > 0:
            perimeter_nodes = self._get_perimeter_nodes(graph, bounds)
            safe_nodes.update(perimeter_nodes)

        return safe_nodes

    def _get_perimeter_nodes(self, graph: nx.MultiDiGraph,
                            bounds: Tuple[float, float, float, float]) -> Set[int]:
        """
        Get nodes at the perimeter of the map (boundary nodes).
        These represent "exits" from the simulation area.
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        perimeter_nodes = set()

        # Define perimeter threshold (5% of map size)
        lon_threshold = (max_lon - min_lon) * 0.05
        lat_threshold = (max_lat - min_lat) * 0.05

        for node, data in graph.nodes(data=True):
            lon, lat = data['x'], data['y']

            # Check if node is near any boundary
            if (lon < min_lon + lon_threshold or  # West edge
                lon > max_lon - lon_threshold or  # East edge
                lat < min_lat + lat_threshold or  # South edge
                lat > max_lat - lat_threshold):   # North edge
                perimeter_nodes.add(node)

        return perimeter_nodes
