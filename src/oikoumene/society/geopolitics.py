"""
Geopolitical Layer — Emergent nation-states, alliances, trade, and conflict.

Nations are NOT pre-defined. They emerge organically when settlements grow
and merge. Alliances, trade networks, and conflicts arise from agent
interactions and macro-level pressures.

References:
- Hughes (2019) IFs / Pardee Center conflict model
- Liberal peace theory (Russett 1993; Oneal & Russett 1999):
  trade interdependence reduces conflict probability
- Earth4All (Dixson-Decleve et al. 2022): social tension -> instability
- Nordhaus DICE: climate damage function
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from oikoumene.society.interactions import InteractionsMixin

if TYPE_CHECKING:
    from oikoumene.macro.macro import MacroState
    from oikoumene.core.world import Settlement


@dataclass
class NationState:
    """
    Emerges when settlements grow large enough or formally merge.
    NOT pre-defined — nations form organically from agent clusters.
    """
    id: int
    name: str
    settlement_ids: list[int] = field(default_factory=list)
    capital_settlement_id: int = 0
    leader_agent_id: Optional[int] = None

    # Aggregate stats (recomputed each tick from member agents)
    population: int = 0
    total_wealth: float = 0.0
    total_military: float = 0.0       # Sum of agents' combat skills
    technology_level: float = 1.0     # Average research skill
    center_lat: float = 0.0
    center_lng: float = 0.0

    # Policy choices
    carbon_policy: float = 0.0        # -1 = max fossil, +1 = max renewable
    trade_openness: float = 0.5       # 0 = autarky, 1 = free trade
    military_spending: float = 0.2
    research_spending: float = 0.1

    # Diplomacy
    alliances: set[int] = field(default_factory=set)
    rivals: set[int] = field(default_factory=set)

    # Climate
    emissions: float = 0.0
    climate_pledge: float = 0.0
    actual_reduction: float = 0.0

    age: int = 0

    # Scenario-anchored nation (e.g. a real country seeded by the present_day
    # scenario). Seeded nations are macro-actors defined by their centre
    # coordinates and seeded aggregate stats; they have no member settlements at
    # init and must NOT be pruned as "empty", nor have their seeded stats zeroed
    # until/unless emergent settlements actually join them.
    seeded: bool = False


# Name generation for emergent nations
_NATION_PREFIXES = [
    "United", "Republic of", "Federation of", "Kingdom of",
    "Commonwealth of", "Free", "People's", "Democratic",
    "Grand", "New", "Greater", "Northern", "Southern",
    "Eastern", "Western", "Central", "Allied",
]
_NATION_ROOTS = [
    "Terra", "Sol", "Vega", "Aria", "Nova", "Eden", "Zion",
    "Atlas", "Orion", "Haven", "Nexus", "Astra", "Ember",
    "Frost", "Jade", "Iron", "Gold", "Silver", "Coral",
    "Storm", "Dawn", "Dusk", "Oak", "Pine", "Stone",
]


def _generate_nation_name(rng: np.random.RandomState) -> str:
    prefix = rng.choice(_NATION_PREFIXES)
    root = rng.choice(_NATION_ROOTS)
    return f"{prefix} {root}"


class GeopoliticalSystem(InteractionsMixin):
    """
    Manages nation-state formation, inter-nation dynamics, trade,
    and conflict resolution.

    Updated every macro tick (not every agent tick) for performance.
    """

    _next_nation_id: int = 0

    # Cap on retained climate-summit records (bounds long-run memory growth).
    _MAX_NEGOTIATION_HISTORY: int = 500

    # Conflict-probability model coefficients (logistic regression on dyad-tick).
    #
    # FIX (v0.2): re-calibrated against UCDP/PRIO active-conflict-prevalence
    # rather than the (poorly-defined) "per dyad-year" target of v0.1. With
    # the macro-update interval of 10 ticks (~10 months), v0.1 produced
    # active-conflict prevalence ~99% in a 5-nation world and ~100% in a
    # 10-nation world over a 75-year BAU run, while UCDP-style prevalence
    # for a small bloc should be ~5-15% (low tension) rising to ~40-70%
    # (high climate stress). New values target:
    #   Friedlich (tens=0.20, 2030): ~5-15%   (Final-C: 8.5%)
    #   Mittel    (tens=0.36, 2050): ~20-40%  (Final-C: 30.8%)
    #   Kritisch  (tens=0.62, 2090): ~40-70%  (Final-C: 63.1%)
    # Calibrated empirically via Monte-Carlo over the dyad RNG; see
    # tests/test_geopolitics.py for the calibration harness.
    #
    # Theoretical anchors retained from v0.1:
    # - Liberal-peace coefficient (TRADE) negative: Russett 1993 / Oneal-Russett 1999
    # - Power-parity (PARITY) positive: Bremer 1992, Dangerous Dyads
    # - Resource competition: Homer-Dixon 1999, environmental scarcity
    # - Alliance restraint negative: Leeds 2003 (ATOP) on conflict suppression
    CONFLICT_BASE_RATE = -7.5         # Logit intercept (raised severity threshold)
    CONFLICT_RESOURCE_COEFF = 2.0     # Resource competition increases conflict
    CONFLICT_PARITY_COEFF = 0.5       # Near-peer powers more likely to fight
    CONFLICT_TRADE_COEFF = -1.5       # Trade interdependence reduces conflict
    CONFLICT_TENSION_COEFF = 1.5      # Social tension amplifies conflict (was 3.0)
    CONFLICT_TERRITORY_COEFF = 1.0    # Territorial overlap increases conflict
    CONFLICT_ALLIANCE_COEFF = -2.0    # Shared alliances reduce conflict
    CONFLICT_HISTORY_COEFF = -0.5     # Positive diplomatic history

    # Minimum thresholds
    NATION_FORMATION_POP = 10         # Min population to form a nation
    NATION_MERGE_DISTANCE = 8.0       # Degrees: settlements close enough to merge

    # --- Conflict-onset realism controls (see _assess_conflicts) -------------
    # Geographic gating: only dyads within this great-circle distance
    # (deg-equivalents, ~111 km/deg) are eligible to ignite. Interstate war is
    # overwhelmingly proximate — Bremer (1992) "Dangerous Dyads"; Vasquez (1993)
    # territorial theory. 20° (~2200 km) keeps the 5-nation calibration cluster
    # (max dyad 11.3°) fully connected while excluding globally-implausible
    # pairings (e.g. Chile–Mongolia) that the ungated all-pairs loop admitted.
    CONFLICT_MAX_DYAD_DEG = 20.0

    # N-aware onset normalization. The per-dyad logistic rate is calibrated on a
    # ~5-nation regional bloc (~10 dyads; test_geopolitics). Ungated, the
    # aggregate onset grew ~linearly with the dyad count — igniting ~12
    # new wars per assessment in the 140-nation present-day world. We hold the
    # aggregate onset roughly constant above a reference number of eligible
    # dyads (global interstate-war onset does not scale with the number of
    # country-pairs). scale = min(1, REFERENCE / n_eligible), so it is exactly
    # 1.0 for any bloc ≤ REFERENCE dyads and the small-N calibration is untouched.
    # Calibrated (present-day, 1178 eligible dyads) to ~1 new interstate-conflict
    # onset/year and a stable ~8–10 active conflicts — in the UCDP interstate/
    # internationalized envelope, versus ~12 new wars per assessment ungated.
    CONFLICT_DYAD_REFERENCE = 500

    # Onset spin-up: a seeded present-day snapshot is a frozen state; new
    # tensions phase in over the first few assessments (~0.83 yr each) instead
    # of igniting maximally at t=0, removing the start-of-run onset transient.
    CONFLICT_SPINUP_ASSESSMENTS = 3

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.nations: list[NationState] = []
        self.relation_graph = nx.DiGraph()   # Weighted edges: +alliance, -rivalry
        self.trade_graph = nx.DiGraph()      # Edge weight = trade volume
        self.active_conflicts: list[dict] = []
        self.negotiation_history: list[dict] = []
        self.rng = rng or np.random.RandomState(42)
        self._conflict_assessments = 0       # for the onset spin-up ramp

    # ------------------------------------------------------------------
    # Main Update (called each macro tick)
    # ------------------------------------------------------------------

    def update(
        self,
        settlements: list,
        agents: list,
        macro_state: 'MacroState',
    ):
        """Full geopolitical update cycle."""
        alive_agents = [a for a in agents if a.alive]

        # 1. Form new nations from large settlements
        self._check_nation_formation(settlements, alive_agents)

        # 2. Update nation aggregate stats
        self._update_nation_stats(settlements, alive_agents)

        # 3. Evolve diplomatic relations
        self._update_relations(macro_state)

        # 4. Resolve trade flows
        self._resolve_trade()

        # 5. Assess and resolve conflicts
        self._assess_conflicts(macro_state)

        # 6. Conduct periodic negotiations
        if len(self.nations) >= 2:
            self._conduct_negotiations(macro_state)

        # 7. Technology diffusion between trading partners
        self._diffuse_technology()

        # Age all nations
        for n in self.nations:
            n.age += 1

    # ------------------------------------------------------------------
    # Nation Formation
    # ------------------------------------------------------------------

    def _check_nation_formation(self, settlements: list, alive_agents: list):
        """Check if settlements should merge into nations."""
        # Settlements already in a nation
        claimed_settlements = set()
        for n in self.nations:
            claimed_settlements.update(n.settlement_ids)

        # Find unclaimed settlements large enough
        unclaimed = [s for s in settlements
                     if s.id not in claimed_settlements
                     and s.population >= self.NATION_FORMATION_POP]

        for settlement in unclaimed:
            # Try to join an existing nearby nation
            joined = False
            for nation in self.nations:
                nation_center = self._nation_center(nation, settlements)
                if nation_center is None:
                    continue
                dist = self._great_circle_deg(
                    settlement.lat, settlement.lng,
                    nation_center[0], nation_center[1],
                )
                if dist < self.NATION_MERGE_DISTANCE:
                    nation.settlement_ids.append(settlement.id)
                    joined = True
                    break

            if not joined:
                # Found a new nation
                GeopoliticalSystem._next_nation_id += 1
                nation = NationState(
                    id=GeopoliticalSystem._next_nation_id,
                    name=_generate_nation_name(self.rng),
                    settlement_ids=[settlement.id],
                    capital_settlement_id=settlement.id,
                    leader_agent_id=settlement.leader_id,
                    center_lat=settlement.lat,
                    center_lng=settlement.lng,
                )
                self.nations.append(nation)
                self.relation_graph.add_node(nation.id)
                self.trade_graph.add_node(nation.id)

        # Remove dead nations (no living settlements). Seeded scenario nations
        # (real countries in present_day) are anchored by their centre coords and
        # seeded stats, so they persist even with zero member settlements.
        living_settlement_ids = {s.id for s in settlements if s.population > 0}
        for nation in self.nations[:]:
            nation.settlement_ids = [
                sid for sid in nation.settlement_ids
                if sid in living_settlement_ids
            ]
            if not nation.settlement_ids and not nation.seeded:
                self.relation_graph.remove_node(nation.id)
                if nation.id in self.trade_graph:
                    self.trade_graph.remove_node(nation.id)
                self.nations.remove(nation)
                # Purge dangling references to the removed nation from every
                # surviving nation's alliance/rival sets, else phantom ids
                # accumulate and inflate the "shared alliances" conflict term.
                for other in self.nations:
                    other.alliances.discard(nation.id)
                    other.rivals.discard(nation.id)

    @staticmethod
    def _great_circle_deg(lat1: float, lng1: float,
                          lat2: float, lng2: float) -> float:
        """
        Great-circle distance expressed in degree-equivalents (km / 111).

        FIX (v0.2): the previous euclidean-in-(lat,lng) distance distorts
        badly at high latitudes — at 60 deg N a "5-degree-distance" spans
        ~280 km west-east versus ~555 km along the equator. Using the
        haversine formula with the radius set so the result is in
        degree-equivalents preserves the existing thresholds (e.g.
        NATION_MERGE_DISTANCE = 8.0 still means ~890 km) without
        requiring re-calibration. At low latitudes the result matches
        euclidean to within ~1%.
        """
        # haversine formula in radians
        phi1 = np.radians(lat1); phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlmb = np.radians(lng2 - lng1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
        c = 2 * np.arcsin(min(1.0, np.sqrt(a)))
        # Earth radius 6371 km, 1 deg of equator ~ 111 km
        return float(c * 6371.0 / 111.0)

    def _nation_center(self, nation: NationState, settlements: list):
        """Compute geographic center of a nation's settlements."""
        coords = []
        for s in settlements:
            if s.id in nation.settlement_ids and s.population > 0:
                coords.append((s.lat, s.lng))
        if not coords:
            # Seeded nations are anchored by their stored centre, so nearby
            # emergent settlements can still be assigned to them by proximity.
            if nation.seeded:
                return (nation.center_lat, nation.center_lng)
            return None
        return (
            np.mean([c[0] for c in coords]),
            np.mean([c[1] for c in coords]),
        )

    # ------------------------------------------------------------------
    # Nation Stats Update
    # ------------------------------------------------------------------

    def _update_nation_stats(self, settlements: list, alive_agents: list):
        """Recompute aggregate nation stats from member agents."""
        # Build settlement -> agents mapping. Index agents by id once (O(agents))
        # and look up each settlement's members directly, so this is
        # O(agents + total_memberships) rather than O(settlements x agents).
        agents_by_id = {a.id: a for a in alive_agents}
        settlement_agents: dict[int, list] = {}
        for s in settlements:
            settlement_agents[s.id] = [
                agents_by_id[aid] for aid in s.members if aid in agents_by_id
            ]

        for nation in self.nations:
            members = []
            for sid in nation.settlement_ids:
                members.extend(settlement_agents.get(sid, []))

            if not members:
                # Seeded scenario nations keep their seeded macro stats
                # (real-world population/wealth) until emergent settlements join.
                if not nation.seeded:
                    nation.population = len(members)  # 0 for empty emergent nation
                continue
            nation.population = len(members)

            nation.total_wealth = max(0.0, sum(a.wealth for a in members))
            nation.total_military = sum(a.skills.get_level("combat") for a in members)
            nation.technology_level = float(np.mean(
                [a.skills.get_level("research") for a in members]
            )) + 1.0  # Base 1.0 + avg research skill

            # Center position
            center = self._nation_center(nation, settlements)
            if center:
                nation.center_lat, nation.center_lng = center

            # Policy: derived from leader agent traits or average
            leader = None
            for a in members:
                if a.id == nation.leader_agent_id:
                    leader = a
                    break
            if leader is None and members:
                # Elect new leader: highest leadership skill
                leader = max(members, key=lambda a: a.skills.get_level("leadership"))
                nation.leader_agent_id = leader.id

            if leader:
                # Carbon policy: high intelligence + curiosity = green policy
                nation.carbon_policy = (
                    leader.traits["intelligence"] * 0.5 +
                    leader.traits["curiosity"] * 0.3 -
                    leader.traits["ambition"] * 0.2 -
                    0.3
                )
                nation.trade_openness = 0.3 + 0.7 * leader.traits["cooperation"]
                nation.military_spending = 0.1 + 0.3 * leader.traits["risk_tolerance"]
                nation.research_spending = 0.1 + 0.3 * leader.traits["intelligence"]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_total_trade_volume(self) -> float:
        """Total trade volume across all edges."""
        return sum(d.get("weight", 0) for _, _, d in self.trade_graph.edges(data=True))

    def get_avg_trust(self) -> float:
        """Average diplomatic relation weight."""
        if self.relation_graph.number_of_edges() == 0:
            return 0.0
        return float(np.mean(
            [d.get("weight", 0) for _, _, d in self.relation_graph.edges(data=True)]
        ))

    def get_conflict_intensity(self) -> float:
        """Total conflict intensity for macro feedback."""
        if not self.active_conflicts:
            return 0.0
        return float(np.mean([c["intensity"] for c in self.active_conflicts]))

    def get_summary(self) -> dict:
        """Summary for UI/stats."""
        return {
            "nations": len(self.nations),
            "active_conflicts": len(self.active_conflicts),
            "trade_volume": round(self.get_total_trade_volume(), 2),
            "avg_trust": round(self.get_avg_trust(), 3),
            "conflict_intensity": round(self.get_conflict_intensity(), 3),
        }

    def get_nations_list(self) -> list[dict]:
        """Serialized nation list for UI."""
        return [
            {
                "id": n.id,
                "name": n.name,
                "population": n.population,
                "wealth": round(n.total_wealth, 1),
                "tech": round(n.technology_level, 3),
                "lat": round(n.center_lat, 2),
                "lng": round(n.center_lng, 2),
                "alliances": list(n.alliances),
                "rivals": list(n.rivals),
                "carbon_policy": round(n.carbon_policy, 2),
                "trade_openness": round(n.trade_openness, 2),
                "climate_pledge": round(n.climate_pledge, 2),
                "age": n.age,
            }
            for n in self.nations
        ]
