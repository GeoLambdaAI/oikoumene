"""
Businesses and settlements (moved from world.py): Business, Settlement, and the
settlements mixin for World with create_business, _find_nearest_settlement and
_check_settlement_formation.

SettlementsMixin expects on the host class: the attributes agents, businesses, rng,
settlements and tick; and the methods _distance_deg() and get_nearby_agents()
(SpatialMixin).
"""

import numpy as np
from typing import Optional


# ============================================================================
# Business / Economy
# ============================================================================

class Business:
    _next_id = 0

    def __init__(self, owner_id: int, lat: float, lng: float,
                 business_type: str, capital: float):
        Business._next_id += 1
        self.id = Business._next_id
        self.owner_id = owner_id
        self.lat = lat
        self.lng = lng
        self.business_type = business_type
        self.capital = capital
        self.revenue = 0.0
        self.employees: list[int] = []
        self.age = 0
        self.reputation = 0.5
        self.active = True

    def operate(self, world) -> float:
        if not self.active:
            return 0.0
        self.age += 1

        workforce = len(self.employees) + 1
        base_revenue = workforce * 2.0 * self.reputation

        if self.business_type in ("farming", "mining", "crafting"):
            resource_type = {"farming": "food", "mining": "minerals",
                             "crafting": "wood"}.get(self.business_type, "food")
            harvested = world.harvest_resource(self.lat, self.lng, resource_type, workforce * 3)
            base_revenue += harvested * 1.5

        revenue = base_revenue * (1.0 + self.capital * 0.001)

        wage_costs = len(self.employees) * 1.5
        operating_costs = 0.5

        profit = revenue - wage_costs - operating_costs
        self.capital += profit
        self.revenue = revenue
        self.reputation = min(1.0, self.reputation + 0.001 * (1 if profit > 0 else -1))

        if self.capital <= 0:
            self.active = False

        return profit

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "lat": round(self.lat, 4),
            "lng": round(self.lng, 4),
            "type": self.business_type,
            "capital": round(self.capital, 1),
            "revenue": round(self.revenue, 1),
            "employees": len(self.employees),
            "age": self.age,
            "reputation": round(self.reputation, 3),
            "active": self.active,
        }


# ============================================================================
# Society / Governance
# ============================================================================

class Settlement:
    """A cluster of agents forming a community."""
    _next_id = 0

    def __init__(self, lat: float, lng: float, founder_id: int,
                 rng: np.random.RandomState):
        self.rng = rng
        Settlement._next_id += 1
        self.id = Settlement._next_id
        self.lat = lat
        self.lng = lng
        self.founder_id = founder_id
        self.members: set[int] = {founder_id}
        self.name = self._generate_name()
        self.population = 1
        self.culture_values: dict[str, float] = {
            "cooperation": 0.5,
            "innovation": 0.5,
            "tradition": 0.5,
            "militarism": 0.2,
            "trade_openness": 0.5,
        }
        self.governance_type = "tribal"
        self.leader_id: Optional[int] = founder_id
        self.laws: list[str] = []
        self.tax_rate = 0.05
        self.treasury = 0.0
        self.age = 0

    def _generate_name(self) -> str:
        prefixes = ["New", "Fort", "Port", "Lake", "Mount", "Green", "Iron",
                     "Gold", "Silver", "Crystal", "Shadow", "Sun", "Star"]
        suffixes = ["haven", "burg", "ton", "dale", "ford", "bridge", "gate",
                     "wood", "field", "peak", "vale", "shore", "hollow"]
        return self.rng.choice(prefixes) + self.rng.choice(suffixes)

    def update(self, agents: list):
        self.age += 1
        living_members = [a for a in agents if a.id in self.members and a.alive]
        self.population = len(living_members)

        # Prune the membership set to living members only. Without this the set
        # grew without bound over long runs (dead agents were never removed),
        # leaking memory and making every O(members) pass (nation stats,
        # geopolitics) progressively slower.
        self.members = {a.id for a in living_members}

        if self.population == 0:
            return

        for trait_name, culture_key in [
            ("cooperation", "cooperation"),
            ("creativity", "innovation"),
            ("risk_tolerance", "militarism"),
        ]:
            avg_trait = np.mean([a.traits[trait_name] for a in living_members])
            self.culture_values[culture_key] = (
                0.95 * self.culture_values[culture_key] + 0.05 * avg_trait
            )

        if self.population >= 20 and self.governance_type == "tribal":
            self.governance_type = "council"
        elif self.population >= 50 and self.governance_type == "council":
            self.governance_type = "republic"
        elif self.population >= 100 and self.governance_type == "republic":
            if self.culture_values["cooperation"] > 0.6:
                self.governance_type = "democracy"

        for agent in living_members:
            tax = agent.wealth * self.tax_rate * 0.01
            agent.wealth -= tax
            self.treasury += tax

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "lat": round(self.lat, 4),
            "lng": round(self.lng, 4),
            "population": self.population,
            "governance": self.governance_type,
            "culture": {k: round(v, 3) for k, v in self.culture_values.items()},
            "leader_id": self.leader_id,
            "treasury": round(self.treasury, 1),
            "tax_rate": self.tax_rate,
            "age": self.age,
        }


class SettlementsMixin:
    # ------------------------------------------------------------------
    # Business Management
    # ------------------------------------------------------------------

    def create_business(self, owner_id: int, lat: float, lng: float,
                        business_type: str, capital: float) -> dict:
        biz = Business(owner_id, lat, lng, business_type, capital)
        self.businesses.append(biz)
        return biz.to_dict()

    # ------------------------------------------------------------------
    # Settlement Management
    # ------------------------------------------------------------------

    def _find_nearest_settlement(self, lat: float, lng: float) -> Optional[Settlement]:
        if not self.settlements:
            return None
        dists = [self._distance_deg(lat, lng, s.lat, s.lng) for s in self.settlements]
        return self.settlements[int(np.argmin(dists))]

    def _check_settlement_formation(self):
        if self.tick % 50 != 0:
            return

        for agent in self.agents:
            if not agent.alive:
                continue
            in_settlement = any(agent.id in s.members for s in self.settlements)
            if in_settlement:
                continue

            nearby = self.get_nearby_agents(agent.lat, agent.lng, 3.0)
            nearby_unaffiliated = [
                a for a in nearby
                if a.id != agent.id
                and not any(a.id in s.members for s in self.settlements)
            ]

            if len(nearby_unaffiliated) >= 4:
                cx = np.mean([a.lat for a in nearby_unaffiliated + [agent]])
                cy = np.mean([a.lng for a in nearby_unaffiliated + [agent]])
                settlement = Settlement(cx, cy, agent.id, rng=self.rng)
                for a in nearby_unaffiliated:
                    settlement.members.add(a.id)
                self.settlements.append(settlement)
