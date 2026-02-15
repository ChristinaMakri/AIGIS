"""
Location-Agnostic Environment Builder with Perlin Noise Terrain
Dynamically identifies safe zones using OSM tags and map perimeter
"""
import numpy as np
import osmnx as ox
import networkx as nx
from typing import Tuple, Dict, List, Set
from shapely.geometry import Point
from noise import pnoise2
import warnings
warnings.filterwarnings('ignore')

from .config import (
    PERLIN_SCALE,
    PERLIN_OCTAVES,
    PERLIN_PERSISTENCE,
    PERLIN_LACUNARITY,
    PERLIN_BASE_HEIGHT,
    PERLIN_AMPLITUDE,
    SAFE_ZONE_TAGS,
    USE_PERIMETER_AS_SAFE
)


class Environment:
    """Container for the simulation environment"""

    def __init__(self, graph: nx.MultiDiGraph, fuel_grid: np.ndarray,
                 obstacle_grid: np.ndarray, elevation_grid: np.ndarray,
                 bounds: Tuple[float, float, float, float],
                 grid_shape: Tuple[int, int],
                 safe_nodes: Set[int]):
        self.graph = graph
        self.fuel_grid = fuel_grid
        self.obstacle_grid = obstacle_grid
        self.elevation_grid = elevation_grid
        self.bounds = bounds  # (min_lon, min_lat, max_lon, max_lat)
        self.grid_shape = grid_shape
        self.safe_nodes = safe_nodes  # Set of node IDs that are safe zones

        # Population density grid (loaded from real OSM data)
        self.population_density = np.zeros(grid_shape)

        # Fire state grids
        self.fire_grid = np.zeros_like(fuel_grid, dtype=np.int8)
        # 0 = no fuel, 1 = burning, 2 = burnt out, 3 = fuel
        self.fire_grid[fuel_grid > 0] = 3

        # Temperature grid (0-100°C, for risk assessment)
        self.temperature_grid = np.zeros_like(fuel_grid, dtype=np.float32)

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
    Generates terrain using Perlin Noise and identifies safe zones dynamically.
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

        # Step 5: Generate Perlin Noise elevation
        print("  ⛰️  Generating Perlin Noise terrain...")
        elevation_grid = self._generate_perlin_terrain()

        # Step 6: Identify safe zones
        print("  🛡️  Identifying safe zones...")
        safe_nodes = self._identify_safe_zones(graph, bounds)

        print(f"✅ Environment built! {len(safe_nodes)} safe zones identified.")

        return Environment(
            graph=graph,
            fuel_grid=fuel_grid,
            obstacle_grid=obstacle_grid,
            elevation_grid=elevation_grid,
            bounds=bounds,
            grid_shape=self.grid_size,
            safe_nodes=safe_nodes
        )

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
            tags = {
                'landuse': ['forest', 'wood'],
                'natural': ['wood', 'tree', 'tree_row']
            }
            gdf = ox.features_from_bbox(
                bbox=(bounds[3], bounds[1], bounds[2], bounds[0]),
                tags=tags
            )
            return list(gdf.geometry)
        except Exception as e:
            print(f"  ⚠️  Could not fetch forests: {e}")
            return []

    def _fetch_buildings(self, bounds: Tuple[float, float, float, float]) -> list:
        """Fetch building geometries from OSM"""
        try:
            tags = {
                'building': True,
                'landuse': ['residential', 'commercial', 'industrial']
            }
            gdf = ox.features_from_bbox(
                bbox=(bounds[3], bounds[1], bounds[2], bounds[0]),
                tags=tags
            )
            return list(gdf.geometry)
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
        Generate realistic terrain using Perlin Noise.
        Returns elevation grid in meters.
        """
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
