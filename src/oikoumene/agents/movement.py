"""
Movement mixin for Agent (agents.py): physics, wandering, separation, goal-directed
movement and resource search.

Expects on the host class: the attributes business_id, energy, goal_lat, goal_lng,
heading_x, heading_y, home_lat, home_lng, id, lat, lng, memory, relationships, rng,
traits, vlat and vlng; the class constants GOAL_FORCE, LAT_MAX, LAT_MIN, MAX_SPEED,
REPRODUCE_MIN_AGE_YEARS, SEPARATION_FORCE, SEPARATION_RADIUS and WANDER_STRENGTH; and
the method _yrs_to_ticks().
"""

from typing import TYPE_CHECKING

import numpy as np

from oikoumene.agents.traits import _wrap_longitude

if TYPE_CHECKING:
    from oikoumene.agents.agents import Agent


class MovementMixin:
    # ------------------------------------------------------------------
    # Physics-Based Movement
    # ------------------------------------------------------------------

    def _apply_physics(self, world):
        """Apply velocity, friction, terrain effects, and boundary handling."""
        from oikoumene.core.earth import is_land

        # Terrain friction modifier
        r, c = world.resources.get_cell(self.lat, self.lng)
        terrain = world.terrain[min(r, world.terrain.shape[0]-1),
                                min(c, world.terrain.shape[1]-1)]
        terrain_friction = {
            0: 0.3,   # Ocean - very slow
            1: 0.90,  # Plains - easy movement
            2: 0.82,  # Forest - some resistance
            3: 0.70,  # Mountains - hard
            4: 0.85,  # Desert - sandy
            5: 0.75,  # Tundra - cold resistance
        }.get(terrain, 0.85)

        # Apply friction
        self.vlat *= terrain_friction
        self.vlng *= terrain_friction

        # Clamp speed — scaled by era (early eras = centuries of migration per tick)
        speed = np.sqrt(self.vlat**2 + self.vlng**2)
        era_mult = getattr(self, '_era_speed', 1.0)
        max_speed = self.MAX_SPEED * era_mult * (0.5 + 0.5 * self.energy / 100.0)
        if speed > max_speed:
            scale = max_speed / (speed + 1e-8)
            self.vlat *= scale
            self.vlng *= scale

        # Apply velocity. Latitude is clamped to the habitable band; longitude
        # wraps at the dateline so land checks near +/-180 use correctly wrapped
        # coordinates instead of reading the wrong grid edge.
        new_lat = float(np.clip(self.lat + self.vlat, self.LAT_MIN, self.LAT_MAX))
        new_lng = _wrap_longitude(self.lng + self.vlng)

        # Ocean avoidance - check if next position is ocean
        if not is_land(new_lat, new_lng):
            # Bounce back and steer away
            self.vlat *= -0.5
            self.vlng *= -0.5
            # Try to find land direction (check 8 directions). Probe coordinates
            # are wrapped/clamped, and the steering offset uses the shortest arc
            # in longitude so a probe across the dateline does not yield a huge
            # spurious vector.
            best_dist = float('inf')
            best_dlat, best_dlng = 0.0, 0.0
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                check_lat = float(np.clip(self.lat + np.sin(angle) * 2.0,
                                          self.LAT_MIN, self.LAT_MAX))
                check_lng = _wrap_longitude(self.lng + np.cos(angle) * 2.0)
                if is_land(check_lat, check_lng):
                    dlat = check_lat - self.lat
                    dlng = _wrap_longitude(check_lng - self.lng)
                    dist = dlat**2 + dlng**2
                    if dist < best_dist:
                        best_dist = dist
                        best_dlat = dlat
                        best_dlng = dlng
            if best_dist < float('inf'):
                d = np.sqrt(best_dlat**2 + best_dlng**2) + 1e-8
                self.vlat += best_dlat / d * 0.03
                self.vlng += best_dlng / d * 0.03
        else:
            self.lat = new_lat
            self.lng = new_lng

        # World boundary: clamp latitude to the habitable band, wrap longitude.
        self.lat = float(np.clip(self.lat, self.LAT_MIN, self.LAT_MAX))
        self.lng = _wrap_longitude(self.lng)

        # Update heading from velocity
        speed = np.sqrt(self.vlat**2 + self.vlng**2)
        if speed > 0.001:
            self.heading_x = self.vlng / speed
            self.heading_y = self.vlat / speed

    def _move_toward(self, tlat: float, tlng: float, strength: float = 1.0):
        """Apply force toward a target position (lat/lng), era-scaled."""
        dlat = tlat - self.lat
        dlng = tlng - self.lng
        dist = np.sqrt(dlat**2 + dlng**2) + 1e-8
        era_mult = getattr(self, '_era_speed', 1.0)
        force = min(strength * era_mult, dist * 0.1 * era_mult)
        self.vlat += (dlat / dist) * force
        self.vlng += (dlng / dist) * force

    def _wander(self):
        """Apply random wandering force for exploration."""
        angle_noise = self.rng.normal(0, 0.3)
        cos_a, sin_a = np.cos(angle_noise), np.sin(angle_noise)
        new_hx = self.heading_x * cos_a - self.heading_y * sin_a
        new_hy = self.heading_x * sin_a + self.heading_y * cos_a
        self.heading_x, self.heading_y = new_hx, new_hy

        era_mult = getattr(self, '_era_speed', 1.0)
        strength = self.WANDER_STRENGTH * (0.5 + 0.5 * self.traits["curiosity"]) * era_mult
        self.vlng += self.heading_x * strength
        self.vlat += self.heading_y * strength

    def _apply_separation(self, nearby_agents: list['Agent']):
        """Keep minimum distance from other agents."""
        for other in nearby_agents:
            if other.id == self.id or not other.alive:
                continue
            dlat = self.lat - other.lat
            dlng = self.lng - other.lng
            dist = np.sqrt(dlat**2 + dlng**2) + 1e-8
            if dist < self.SEPARATION_RADIUS:
                push = self.SEPARATION_FORCE * (1.0 - dist / self.SEPARATION_RADIUS)
                self.vlat += (dlat / dist) * push
                self.vlng += (dlng / dist) * push

    def _apply_goal_movement(self, goal: str, action_vector: np.ndarray, world):
        """Apply movement forces based on the current goal."""
        # Always wander a bit
        self._wander()

        # CEM-derived movement direction
        cem_dlat = float(np.tanh(action_vector[0]))
        cem_dlng = float(np.tanh(action_vector[1]))
        cem_strength = 0.015 + 0.015 * float(np.clip(abs(action_vector[2]), 0, 1))
        self.vlat += cem_dlat * cem_strength
        self.vlng += cem_dlng * cem_strength

        # Goal-specific movement
        if goal == "eat":
            best = self._find_resource_direction(world, "food")
            if best is not None:
                self._move_toward(best[0], best[1], self.GOAL_FORCE * 1.5)
                self.goal_lat, self.goal_lng = best

        elif goal in ("socialize", "reproduce", "trade"):
            nearby = world.get_nearby_agents(self.lat, self.lng, 10.0)
            others = [a for a in nearby if a.id != self.id and a.alive]
            if others:
                if goal == "reproduce":
                    min_age = self._yrs_to_ticks(self.REPRODUCE_MIN_AGE_YEARS)
                    candidates = [a for a in others if a.energy > 30 and a.age > min_age
                                  and a.reproduction_cooldown <= 0]
                    if candidates:
                        target = max(candidates,
                                     key=lambda a: self.relationships.get(a.id, 0))
                    else:
                        target = min(others,
                                     key=lambda a: (a.lat-self.lat)**2 + (a.lng-self.lng)**2)
                else:
                    target = min(others,
                                 key=lambda a: (a.lat-self.lat)**2 + (a.lng-self.lng)**2)
                self._move_toward(target.lat, target.lng, self.GOAL_FORCE)
                self.goal_lat, self.goal_lng = target.lat, target.lng

        elif goal == "explore":
            if self.rng.random() < 0.05:
                angle = self.rng.uniform(0, 2 * np.pi)
                self.heading_x = np.cos(angle)
                self.heading_y = np.sin(angle)
            self.vlng += self.heading_x * self.GOAL_FORCE * self.traits["curiosity"]
            self.vlat += self.heading_y * self.GOAL_FORCE * self.traits["curiosity"]

        elif goal in ("work", "build_business"):
            if self.home_lat is not None:
                self._move_toward(self.home_lat, self.home_lng, self.GOAL_FORCE * 0.5)
            elif self.business_id is not None:
                for biz in world.businesses:
                    if biz.id == self.business_id:
                        self._move_toward(biz.lat, biz.lng, self.GOAL_FORCE)
                        break

        elif goal == "heal":
            self.vlat *= 0.5
            self.vlng *= 0.5

        elif goal == "govern":
            settlement = world._find_nearest_settlement(self.lat, self.lng)
            if settlement:
                self._move_toward(settlement.lat, settlement.lng, self.GOAL_FORCE)

        elif goal == "migrate":
            # Strong movement in current heading (fleeing)
            self.vlng += self.heading_x * self.GOAL_FORCE * 2.0
            self.vlat += self.heading_y * self.GOAL_FORCE * 2.0

    def _find_resource_direction(self, world, resource_type: str):
        """Scan nearby cells for best resource location."""
        best_val = 0
        best_pos = None
        cr, cc = world.resources.get_cell(self.lat, self.lng)
        csd = world.resources.cell_size_deg

        # Scan 5x5 area around agent
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r = cr + dr
                c = cc + dc
                if 0 <= r < world.resources.rows and 0 <= c < world.resources.cols:
                    layer = getattr(world.resources, resource_type)
                    val = layer[r, c]
                    if val > best_val:
                        best_val = val
                        # Convert grid cell back to lat/lng
                        cell_lat = world.resources.lat_max - (r + 0.5) * csd
                        cell_lng = world.resources.lng_min + (c + 0.5) * csd
                        best_pos = (cell_lat, cell_lng)

        if best_val > 1.0:
            if best_pos:
                angle = np.arctan2(best_pos[0] - self.lat, best_pos[1] - self.lng)
                self.memory.learn_fact("best_food_direction", angle)
            return best_pos
        return None
