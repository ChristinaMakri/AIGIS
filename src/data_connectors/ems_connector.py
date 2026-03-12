"""
Emergency Medical Services (EMS) Data Connector
Uses OpenStreetMap via osmnx — free, no API key required.

Fetches from OSM:
  - Hospitals (amenity=hospital)
  - Ambulance stations (emergency=ambulance_station)
  - Fire stations (amenity=fire_station) — used for firefighter pre-positioning

Returns road-network node IDs so agents can route to these facilities
using the existing NetworkX graph already built by LiveMapBuilder.
"""
from typing import List, Dict, Tuple, Optional

try:
    import osmnx as ox
    _OSMNX_AVAILABLE = True
except ImportError:
    _OSMNX_AVAILABLE = False


class EMSConnector:
    """
    Queries OSM for emergency facilities and maps them to road-network nodes.

    The connector returns node IDs from the same graph used by all agents,
    so AmbulanceAgent can route to them with nx.shortest_path directly.
    """

    def __init__(self):
        self._cache: Optional[List[Dict]] = None

    def fetch_facilities(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        graph,
    ) -> List[Dict]:
        """
        Fetch emergency facilities near (lat, lon) within radius_m.

        Args:
            lat, lon:  Centre coordinate
            radius_m:  Search radius in metres
            graph:     NetworkX road graph (from environment.graph)

        Returns:
            List of dicts, each with:
                type (str), name (str), lat (float), lon (float),
                node_id (int) — nearest road-network node
        """
        if self._cache is not None:
            return self._cache

        if not _OSMNX_AVAILABLE:
            print("  [EMS] osmnx not available — no facility data.")
            return []

        facilities = []
        tag_sets = [
            ("hospital",          {"amenity": "hospital"}),
            ("ambulance_station", {"emergency": "ambulance_station"}),
            ("fire_station",      {"amenity": "fire_station"}),
        ]

        for fac_type, tags in tag_sets:
            try:
                gdf = ox.features_from_point((lat, lon), tags=tags, dist=int(radius_m))
                for _, row in gdf.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    # Use centroid for both point and polygon features
                    centroid = geom.centroid if geom.geom_type != "Point" else geom
                    f_lat, f_lon = centroid.y, centroid.x

                    try:
                        node_id = ox.distance.nearest_nodes(graph, f_lon, f_lat)
                    except Exception:
                        continue

                    name = row.get("name", fac_type.replace("_", " ").title())
                    if not isinstance(name, str):
                        name = fac_type.replace("_", " ").title()

                    facilities.append({
                        "type":    fac_type,
                        "name":    name,
                        "lat":     f_lat,
                        "lon":     f_lon,
                        "node_id": node_id,
                    })
            except Exception as e:
                # Tag may not exist in the area — silently skip
                pass

        if facilities:
            counts = {}
            for f in facilities:
                counts[f["type"]] = counts.get(f["type"], 0) + 1
            summary = ", ".join(f"{v} {k}" for k, v in counts.items())
            print(f"  [EMS] Found: {summary}")
        else:
            print("  [EMS] No emergency facilities found in OSM for this area.")

        self._cache = facilities
        return facilities

    def hospital_nodes(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        graph,
    ) -> List[int]:
        """
        Return road-network node IDs for hospitals only.
        Falls back to an empty list (Ambulance will use safe zones instead).
        """
        facilities = self.fetch_facilities(lat, lon, radius_m, graph)
        nodes = [f["node_id"] for f in facilities if f["type"] == "hospital"]
        return nodes

    def firefighter_station_positions(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        graph,
    ) -> List[Tuple[float, float]]:
        """
        Return (lat, lon) tuples for fire stations.
        Used to spawn Firefighter agents at real OSM fire station locations.
        """
        facilities = self.fetch_facilities(lat, lon, radius_m, graph)
        return [(f["lat"], f["lon"]) for f in facilities if f["type"] == "fire_station"]
