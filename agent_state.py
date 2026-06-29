"""
Vectorized Agent State — Structure of Arrays for batch processing.

All numerical agent state lives in contiguous numpy arrays, enabling
BLAS-accelerated batch operations. Individual Agent objects hold indices
into these arrays plus non-vectorizable state (memory, relationships).

Memory layout: "Hot state in arrays, cold state in objects."
Reference: Noel Llopis, "Data-Oriented Design" (2009).

STATUS: benchmarked building block, NOT yet wired into the live engine (see
README "Future SoA Building Blocks"). The live per-agent physics lives in
``agents.py``. The batch constants here (e.g. the 8000-tick metabolism
denominator and the 800-tick aging threshold in ``batch_metabolism``) are the
*Modern-era* (1 month/tick) equivalents of the era-aware ``_yrs_to_ticks(...)``
thresholds in ``Agent.update``; if this path is ever integrated, those must be
made era-aware to match ``agents.py`` exactly.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Optional


# Trait indices
T_INTELLIGENCE = 0
T_CREATIVITY = 1
T_SOCIABILITY = 2
T_AMBITION = 3
T_RISK_TOLERANCE = 4
T_COOPERATION = 5
T_RESILIENCE = 6
T_CURIOSITY = 7
N_TRAITS = 8

# Skill indices
S_FARMING = 0
S_MINING = 1
S_CRAFTING = 2
S_TRADING = 3
S_BUILDING = 4
S_RESEARCH = 5
S_LEADERSHIP = 6
S_DIPLOMACY = 7
S_COMBAT = 8
S_MEDICINE = 9
N_SKILLS = 10

# Action indices
A_EAT = 0
A_WORK = 1
A_TRADE = 2
A_BUILD = 3
A_SOCIALIZE = 4
A_REPRODUCE = 5
A_EXPLORE = 6
A_RESEARCH = 7
A_HEAL = 8
A_GOVERN = 9
A_MIGRATE = 10
N_ACTIONS = 11

OBS_DIM = 40
ACTION_DIM = 8
LATENT_DIM = 24


class AgentStateArrays:
    """
    Contiguous array storage for all vectorizable agent state.
    Active agents tracked via alive mask. Slot reuse via free list.
    """

    def __init__(self, max_agents: int = 4096):
        self.max_agents = max_agents
        self.count = 0  # High-water mark of slots used

        # --- Hot state (float32 for cache efficiency) ---
        self.lat = np.zeros(max_agents, dtype=np.float32)
        self.lng = np.zeros(max_agents, dtype=np.float32)
        self.vlat = np.zeros(max_agents, dtype=np.float32)
        self.vlng = np.zeros(max_agents, dtype=np.float32)

        self.energy = np.zeros(max_agents, dtype=np.float32)
        self.health = np.zeros(max_agents, dtype=np.float32)
        self.wealth = np.zeros(max_agents, dtype=np.float32)
        self.happiness = np.zeros(max_agents, dtype=np.float32)

        self.age = np.zeros(max_agents, dtype=np.int32)
        self.generation = np.zeros(max_agents, dtype=np.int32)
        self.alive = np.zeros(max_agents, dtype=np.bool_)
        self.reproduction_cooldown = np.zeros(max_agents, dtype=np.int32)

        # Nation membership (-1 = unaffiliated)
        self.nation_id = np.full(max_agents, -1, dtype=np.int32)

        # --- Traits & skills ---
        self.traits = np.zeros((max_agents, N_TRAITS), dtype=np.float32)
        self.skills = np.zeros((max_agents, N_SKILLS), dtype=np.float32)

        # --- JEPA batch buffers ---
        self.observations = np.zeros((max_agents, OBS_DIM), dtype=np.float32)
        self.latent_z = np.zeros((max_agents, LATENT_DIM), dtype=np.float32)
        self.actions = np.zeros((max_agents, ACTION_DIM), dtype=np.float32)

        # --- Action/goal indices ---
        self.current_action_idx = np.zeros(max_agents, dtype=np.int32)
        self.current_goal_idx = np.zeros(max_agents, dtype=np.int32)

        # --- Movement ---
        self.heading_x = np.zeros(max_agents, dtype=np.float32)
        self.heading_y = np.zeros(max_agents, dtype=np.float32)

        # --- Free list for slot reuse ---
        self._free_slots: list[int] = []

        # --- Caches ---
        self._kdtree: Optional[cKDTree] = None
        self._alive_indices: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def add_agent(self, lat: float, lng: float,
                  traits: np.ndarray, skills: np.ndarray,
                  energy: float = 100.0, health: float = 100.0,
                  wealth: float = 10.0, generation: int = 0) -> int:
        """Add a new agent. Returns slot index. Reuses dead slots."""
        if self._free_slots:
            idx = self._free_slots.pop()
        elif self.count < self.max_agents:
            idx = self.count
            self.count += 1
        else:
            return -1  # Buffer full

        self.lat[idx] = lat
        self.lng[idx] = lng
        self.vlat[idx] = np.random.uniform(-0.02, 0.02)
        self.vlng[idx] = np.random.uniform(-0.02, 0.02)
        self.energy[idx] = energy
        self.health[idx] = health
        self.wealth[idx] = wealth
        self.happiness[idx] = 50.0
        self.age[idx] = 0
        self.generation[idx] = generation
        self.alive[idx] = True
        self.reproduction_cooldown[idx] = 0
        self.nation_id[idx] = -1
        self.traits[idx] = traits[:N_TRAITS]
        self.skills[idx] = skills[:N_SKILLS]
        self.current_action_idx[idx] = A_EXPLORE
        self.current_goal_idx[idx] = A_EAT

        angle = np.random.uniform(0, 2 * np.pi)
        self.heading_x[idx] = np.cos(angle)
        self.heading_y[idx] = np.sin(angle)

        self._alive_indices = None  # Invalidate
        return idx

    def kill_agent(self, idx: int):
        self.alive[idx] = False
        self._free_slots.append(idx)
        self._alive_indices = None

    def get_alive_indices(self) -> np.ndarray:
        if self._alive_indices is None:
            self._alive_indices = np.where(self.alive[:self.count])[0]
        return self._alive_indices

    def invalidate_caches(self):
        self._alive_indices = None
        self._kdtree = None

    @property
    def n_alive(self) -> int:
        return len(self.get_alive_indices())

    # ------------------------------------------------------------------
    # Spatial queries (cKDTree)
    # ------------------------------------------------------------------

    def rebuild_kdtree(self):
        """Build cKDTree from alive agent positions. O(N log N)."""
        idx = self.get_alive_indices()
        if len(idx) == 0:
            self._kdtree = None
            return
        positions = np.column_stack([self.lat[idx], self.lng[idx]])
        self._kdtree = cKDTree(positions)

    def query_nearby(self, lat: float, lng: float, radius: float) -> np.ndarray:
        """Find alive agents within radius degrees. Returns global indices."""
        if self._kdtree is None:
            return np.array([], dtype=np.int64)
        local = self._kdtree.query_ball_point([lat, lng], radius)
        return self.get_alive_indices()[local]

    def query_all_neighbors(self, radius: float) -> list:
        """Batch neighbor query for all alive agents. Returns list of local index arrays."""
        if self._kdtree is None:
            return []
        return self._kdtree.query_ball_tree(self._kdtree, radius)

    # ------------------------------------------------------------------
    # Batch physics operations
    # ------------------------------------------------------------------

    def batch_metabolism(self, era_speed: float = 1.0):
        """Vectorized metabolism for all alive agents."""
        idx = self.get_alive_indices()
        if len(idx) == 0:
            return

        # Age and cooldown
        self.age[idx] += 1
        self.reproduction_cooldown[idx] = np.maximum(0, self.reproduction_cooldown[idx] - 1)

        # Energy drain. 8000 ticks = the Modern-era equivalent of Agent.update's
        # era-aware METABOLISM_DOUBLE_YEARS (8000 ticks * 1/12 yr/tick ~= 666 yr).
        base = 0.15 + self.age[idx].astype(np.float32) / 8000.0
        speed = np.sqrt(self.vlat[idx]**2 + self.vlng[idx]**2)
        self.energy[idx] -= base + speed * 0.5
        self.happiness[idx] *= 0.999

        # Health decay for old agents. 800 ticks = Modern-era equivalent of
        # Agent.update's era-aware AGING_THRESHOLD_YEARS (~66.7 yr * 12 ticks/yr).
        old = idx[self.age[idx] > 800]
        if len(old) > 0:
            self.health[old] -= 0.03 * (self.age[old].astype(np.float32) / 800.0)
            self.health[old] += 0.01 * self.traits[old, T_RESILIENCE]

    def batch_death_check(self) -> np.ndarray:
        """Check for dead agents. Returns indices of newly dead."""
        idx = self.get_alive_indices()
        dead_mask = (self.energy[idx] <= 0) | (self.health[idx] <= 0)
        dead = idx[dead_mask]
        for d in dead:
            self.kill_agent(d)
        return dead

    def batch_apply_physics(self, landmask_func, era_speed: float = 1.0):
        """
        Vectorized position/velocity update for all alive agents.

        landmask_func: callable(lat_array, lng_array) -> bool_array
        """
        idx = self.get_alive_indices()
        if len(idx) == 0:
            return

        max_speed = 0.15 * min(20.0, era_speed)

        # Friction
        self.vlat[idx] *= 0.85
        self.vlng[idx] *= 0.85

        # Clamp speed
        speed = np.sqrt(self.vlat[idx]**2 + self.vlng[idx]**2)
        energy_factor = 0.5 + 0.5 * self.energy[idx] / 100.0
        max_s = max_speed * energy_factor
        too_fast = speed > max_s
        if too_fast.any():
            fast_idx = idx[too_fast]
            scale = max_s[too_fast] / (speed[too_fast] + 1e-8)
            self.vlat[fast_idx] *= scale
            self.vlng[fast_idx] *= scale

        # Propose new positions
        new_lat = self.lat[idx] + self.vlat[idx]
        new_lng = self.lng[idx] + self.vlng[idx]

        # Ocean avoidance: check which new positions are on ocean
        on_land = landmask_func(new_lat, new_lng)
        on_ocean = ~on_land

        if on_ocean.any():
            ocean_idx = idx[on_ocean]
            self.vlat[ocean_idx] *= -0.5
            self.vlng[ocean_idx] *= -0.5
            # Don't update position for ocean agents
        if on_land.any():
            land_idx = idx[on_land]
            self.lat[land_idx] = new_lat[on_land]
            self.lng[land_idx] = new_lng[on_land]

        # Boundary clamp
        np.clip(self.lat[idx], -58, 73, out=self.lat[idx])
        np.clip(self.lng[idx], -178, 178, out=self.lng[idx])

        # Update heading from velocity
        speed = np.sqrt(self.vlat[idx]**2 + self.vlng[idx]**2)
        moving = speed > 0.001
        if moving.any():
            m_idx = idx[moving]
            self.heading_x[m_idx] = self.vlng[m_idx] / speed[moving]
            self.heading_y[m_idx] = self.vlat[m_idx] / speed[moving]

    def batch_wander(self, era_speed: float = 1.0):
        """Apply random wandering force to all alive agents."""
        idx = self.get_alive_indices()
        if len(idx) == 0:
            return

        strength = 0.02 * (0.5 + 0.5 * self.traits[idx, T_CURIOSITY]) * min(20.0, era_speed)

        # Rotate heading with noise
        angle_noise = np.random.normal(0, 0.3, size=len(idx)).astype(np.float32)
        cos_a = np.cos(angle_noise)
        sin_a = np.sin(angle_noise)
        new_hx = self.heading_x[idx] * cos_a - self.heading_y[idx] * sin_a
        new_hy = self.heading_x[idx] * sin_a + self.heading_y[idx] * cos_a
        self.heading_x[idx] = new_hx
        self.heading_y[idx] = new_hy

        self.vlng[idx] += new_hx * strength
        self.vlat[idx] += new_hy * strength

    def batch_separation(self, neighbor_lists: list, radius: float = 0.3,
                          force: float = 0.04):
        """Vectorized separation from nearby agents."""
        idx = self.get_alive_indices()
        if len(idx) == 0:
            return

        for i, local_neighbors in enumerate(neighbor_lists):
            if len(local_neighbors) <= 1:
                continue
            gi = idx[i]
            for j in local_neighbors:
                if j == i:
                    continue
                gj = idx[j]
                dlat = self.lat[gi] - self.lat[gj]
                dlng = self.lng[gi] - self.lng[gj]
                dist = np.sqrt(dlat**2 + dlng**2) + 1e-8
                if dist < radius:
                    push = force * (1.0 - dist / radius)
                    self.vlat[gi] += (dlat / dist) * push
                    self.vlng[gi] += (dlng / dist) * push

    def batch_population_pressure(self, neighbor_counts: np.ndarray,
                                    neighbor_centers_lat: np.ndarray,
                                    neighbor_centers_lng: np.ndarray,
                                    era_speed: float = 1.0):
        """Push agents outward from crowded areas."""
        idx = self.get_alive_indices()
        if len(idx) == 0:
            return

        crowded = neighbor_counts > 3
        if not crowded.any():
            return

        c_idx = idx[crowded]
        pressure = 0.05 * (neighbor_counts[crowded] - 3) * min(20.0, era_speed)
        dlat = self.lat[c_idx] - neighbor_centers_lat[crowded]
        dlng = self.lng[c_idx] - neighbor_centers_lng[crowded]
        dist = np.sqrt(dlat**2 + dlng**2) + 1e-8
        self.vlat[c_idx] += (dlat / dist) * pressure
        self.vlng[c_idx] += (dlng / dist) * pressure
