"""
Spatial mixin for World (world.py): the unit-sphere cKDTree neighbour index,
neighbour queries, local state, resource harvesting and the great-circle
_distance_deg; plus the module helpers _unit_sphere_xyz and _chord_for_radius_deg.

Expects on the host class: the attributes _kdtree, _kdtree_alive, agents, bridge,
elevation, fertility, geopolitics, macro, resources and terrain; and the method
_find_nearest_settlement() (SettlementsMixin).
"""

import numpy as np

# Neighbour search radii are in degree-equivalents, the same unit as
# World._distance_deg: 1 deg = 111 km on a sphere of radius 6371 km.
_KM_PER_DEG_EQUIV = 111.0
_EARTH_RADIUS_KM = 6371.0


def _unit_sphere_xyz(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """(lat, lng) in degrees -> (N, 3) unit-sphere coordinates."""
    phi, lam = np.radians(lat), np.radians(lng)
    return np.column_stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])


def _chord_for_radius_deg(radius_deg: float) -> float:
    """Chord length on the unit sphere equal to a great-circle radius in degree-equivalents."""
    angle = radius_deg * _KM_PER_DEG_EQUIV / _EARTH_RADIUS_KM
    return float(2.0 * np.sin(angle / 2.0))


class SpatialMixin:
    def _rebuild_spatial_grid(self):
        """Rebuild cKDTree on unit-sphere xyz of alive agents. O(N log N).

        Indexing 3D unit vectors (not raw lat/lng degrees) makes the radius
        query a true great-circle query: no shrinkage of east-west distances
        with latitude and no break at the 180 deg meridian.
        """
        from scipy.spatial import cKDTree
        alive = [a for a in self.agents if a.alive]
        self._kdtree_alive = alive
        if alive:
            lat = np.array([a.lat for a in alive])
            lng = np.array([a.lng for a in alive])
            self._kdtree = cKDTree(_unit_sphere_xyz(lat, lng))
        else:
            self._kdtree = None

    def get_nearby_agents(self, lat: float, lng: float, radius: float) -> list:
        """Alive agents within a great-circle ``radius`` (degree-equivalents, as
        _distance_deg), from positions at the last spatial-grid rebuild.
        Returned in agent-list order. O(log N) per query."""
        if self._kdtree is None or not self._kdtree_alive:
            return []
        point = _unit_sphere_xyz(np.array([lat]), np.array([lng]))[0]
        indices = sorted(self._kdtree.query_ball_point(point, _chord_for_radius_deg(radius)))
        return [self._kdtree_alive[i] for i in indices]

    def get_local_state(self, lat: float, lng: float) -> dict:
        """Get the local world state visible to an agent."""
        resources = self.resources.get_local(lat, lng)
        nearby = self.get_nearby_agents(lat, lng, 5.0)  # ~5 degrees radius
        r, c = self.resources.get_cell(lat, lng)
        r = min(r, self.terrain.shape[0] - 1)
        c = min(c, self.terrain.shape[1] - 1)

        avg_wealth = np.mean([a.wealth for a in nearby]) if nearby else 0
        social_trust = np.mean([a.traits["cooperation"] for a in nearby]) if nearby else 0.5

        settlement = self._find_nearest_settlement(lat, lng)
        gov_stability = 0.5
        if settlement and self._distance_deg(lat, lng, settlement.lat, settlement.lng) < 5.0:
            gov_stability = min(1.0, settlement.age / 200.0 + 0.3)

        # Nearest agent direction
        nearest_dx, nearest_dy = 0.0, 0.0
        others = [a for a in nearby if self._distance_deg(lat, lng, a.lat, a.lng) > 0.01]
        if others:
            nearest = min(others, key=lambda a: self._distance_deg(lat, lng, a.lat, a.lng))
            dist = self._distance_deg(lat, lng, nearest.lat, nearest.lng)
            if dist > 0:
                nearest_dx = (nearest.lng - lng) / dist
                nearest_dy = (nearest.lat - lat) / dist

        state = {
            "local_food": resources["food"],
            "local_minerals": resources["minerals"],
            "local_wood": resources["wood"],
            "local_water": resources["water"],
            "nearby_agents": len(nearby),
            "local_demand": max(0, len(nearby) * 2 - resources["food"]) / 50.0,
            "local_supply": sum(resources.values()) / 200.0,
            "avg_wealth": avg_wealth,
            "terrain_type": int(self.terrain[r, c]),
            "fertility": float(self.fertility[r, c]),
            "elevation": float(self.elevation[r, c]),
            "social_trust": social_trust,
            "governance_stability": gov_stability,
            "nearest_agent_dx": nearest_dx,
            "nearest_agent_dy": nearest_dy,
        }

        # Add macro-derived signals for agent observations
        macro_state = self.bridge.get_macro_local_state(
            self.macro.state, lat, lng, self.geopolitics, self
        )
        state.update(macro_state)

        return state

    def harvest_resource(self, lat: float, lng: float, resource_type: str, amount: float) -> float:
        return self.resources.harvest(lat, lng, resource_type, amount)

    @staticmethod
    def _distance_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        Great-circle distance expressed in degree-equivalents (km / 111).

        FIX (v0.2): the previous euclidean-in-(lat,lng) formula distorts
        badly at high latitudes — at 60 deg N a "5-degree-distance" along
        longitude spans only ~280 km versus ~555 km along the equator.
        The simulation runs across lat -60..75, so this matters for any
        nearby-agent or settlement-proximity check above ~30 deg latitude.

        We use the haversine formula with units chosen so the result is
        in degree-equivalents, preserving all existing thresholds (e.g.
        "settlement within 5 degrees" still means ~555 km regardless of
        latitude). At low latitudes the result matches the previous
        euclidean approximation to within ~1%.

        Note: get_nearby_agents uses the same metric (unit-sphere cKDTree
        with the chord equivalent of the great-circle radius), so the
        neighbour sets and this distance agree.
        """
        phi1 = np.radians(lat1); phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlmb = np.radians(lng2 - lng1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
        c = 2 * np.arcsin(min(1.0, np.sqrt(a)))
        # Earth radius 6371 km, 1 deg of equator ~ 111 km
        return float(c * 6371.0 / 111.0)
