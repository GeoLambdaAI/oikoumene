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

if TYPE_CHECKING:
    from macro import MacroState
    from world import Settlement


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


class GeopoliticalSystem:
    """
    Manages nation-state formation, inter-nation dynamics, trade,
    and conflict resolution.

    Updated every macro tick (not every agent tick) for performance.
    """

    _next_nation_id: int = 0

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

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.nations: list[NationState] = []
        self.relation_graph = nx.DiGraph()   # Weighted edges: +alliance, -rivalry
        self.trade_graph = nx.DiGraph()      # Edge weight = trade volume
        self.active_conflicts: list[dict] = []
        self.negotiation_history: list[dict] = []
        self.rng = rng or np.random.RandomState(42)

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

        # Remove dead nations (no living settlements)
        living_settlement_ids = {s.id for s in settlements if s.population > 0}
        for nation in self.nations[:]:
            nation.settlement_ids = [
                sid for sid in nation.settlement_ids
                if sid in living_settlement_ids
            ]
            if not nation.settlement_ids:
                self.relation_graph.remove_node(nation.id)
                if nation.id in self.trade_graph:
                    self.trade_graph.remove_node(nation.id)
                self.nations.remove(nation)

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

            nation.population = len(members)
            if not members:
                continue

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
    # Diplomatic Relations
    # ------------------------------------------------------------------

    def _update_relations(self, macro_state: 'MacroState'):
        """Evolve alliance/rivalry graph between nations."""
        for i, na in enumerate(self.nations):
            for j, nb in enumerate(self.nations):
                if i >= j:
                    continue

                # Current relation weight (-1 = rivals, +1 = allies)
                current = 0.0
                if self.relation_graph.has_edge(na.id, nb.id):
                    current = self.relation_graph[na.id][nb.id].get("weight", 0.0)

                # Factors that improve relations
                trade_bonus = 0.0
                if self.trade_graph.has_edge(na.id, nb.id):
                    trade_bonus = 0.01 * self.trade_graph[na.id][nb.id].get("weight", 0)

                # Similar carbon policy = trust
                policy_alignment = 1.0 - abs(na.carbon_policy - nb.carbon_policy)
                alignment_bonus = 0.005 * policy_alignment

                # Factors that worsen relations
                # Resource competition (close + both extractive)
                dist = self._great_circle_deg(
                    na.center_lat, na.center_lng,
                    nb.center_lat, nb.center_lng,
                )
                proximity_tension = max(0, 0.01 * (10.0 - dist) / 10.0)

                # Global tension amplifies rivalry
                tension_factor = 0.005 * macro_state.social_tension

                # Power parity increases rivalry (Thucydides trap)
                if na.population > 0 and nb.population > 0:
                    power_a = na.total_wealth + na.total_military * 10
                    power_b = nb.total_wealth + nb.total_military * 10
                    parity = float(np.clip(
                        1.0 - abs(power_a - power_b) / (max(power_a, power_b) + 1),
                        0.0, 1.0))
                    parity_tension = 0.003 * parity
                else:
                    parity_tension = 0.0

                # Update relation
                delta = (
                    trade_bonus + alignment_bonus -
                    proximity_tension - tension_factor - parity_tension
                )
                new_weight = np.clip(current + delta, -1.0, 1.0)

                self.relation_graph.add_edge(na.id, nb.id, weight=new_weight)
                self.relation_graph.add_edge(nb.id, na.id, weight=new_weight)

                # Update alliance/rivalry sets
                if new_weight > 0.3:
                    na.alliances.add(nb.id)
                    nb.alliances.add(na.id)
                    na.rivals.discard(nb.id)
                    nb.rivals.discard(na.id)
                elif new_weight < -0.3:
                    na.rivals.add(nb.id)
                    nb.rivals.add(na.id)
                    na.alliances.discard(nb.id)
                    nb.alliances.discard(na.id)
                else:
                    na.alliances.discard(nb.id)
                    nb.alliances.discard(na.id)
                    na.rivals.discard(nb.id)
                    nb.rivals.discard(na.id)

    # ------------------------------------------------------------------
    # Trade Network
    # ------------------------------------------------------------------

    def _resolve_trade(self):
        """Update bilateral trade volumes based on openness and relations."""
        for i, na in enumerate(self.nations):
            for j, nb in enumerate(self.nations):
                if i >= j:
                    continue

                if na.population == 0 or nb.population == 0:
                    continue

                # Trade volume = f(openness, relations, economic size)
                openness = na.trade_openness * nb.trade_openness
                relation = 0.5  # Neutral
                if self.relation_graph.has_edge(na.id, nb.id):
                    relation = (self.relation_graph[na.id][nb.id]["weight"] + 1.0) / 2.0

                economic_mass = max(0.0, na.total_wealth * nb.total_wealth) ** 0.5
                distance = self._great_circle_deg(
                    na.center_lat, na.center_lng,
                    nb.center_lat, nb.center_lng,
                )
                # Gravity model: trade proportional to mass, inversely to distance
                # Source: Tinbergen 1962, gravity model of trade
                gravity = economic_mass / max(1.0, distance)

                volume = openness * relation * gravity * 0.01
                volume = max(0, volume)

                if volume > 0.01:
                    self.trade_graph.add_edge(na.id, nb.id, weight=volume)
                    self.trade_graph.add_edge(nb.id, na.id, weight=volume)
                else:
                    # FIX B3: dyad fell below the retention threshold this
                    # tick. Without explicit removal, edges added in earlier
                    # ticks persist with stale weights and feed phantom
                    # values into _diffuse_technology, conflict_probability
                    # (trade_interdep term), and _update_relations (trade_bonus).
                    if self.trade_graph.has_edge(na.id, nb.id):
                        self.trade_graph.remove_edge(na.id, nb.id)
                    if self.trade_graph.has_edge(nb.id, na.id):
                        self.trade_graph.remove_edge(nb.id, na.id)

    # ------------------------------------------------------------------
    # Conflict Assessment
    # ------------------------------------------------------------------

    def _assess_conflicts(self, macro_state: 'MacroState'):
        """
        Evaluate and resolve inter-nation conflicts.

        Conflict probability model adapted from IFs / Pardee Center (Hughes 2019).
        """
        # Decay existing conflicts.
        # FIX (v0.2): previous decay=0.95 with cutoff 0.05 yielded effective
        # conflict duration of ~45 ticks (~38 years) — far longer than UCDP/PRIO
        # median active-conflict duration of ~3 years (Pettersson 2024). The
        # new decay=0.80 produces a ~2.6-year half-life and ~25-tick max duration,
        # consistent with the UCDP record. Combined with the conflict-onset
        # re-calibration this brings active-conflict prevalence into the
        # target envelope of ~10-25% / 30-50% / 55-80% across BAU 2030/2050/2090.
        for conflict in self.active_conflicts[:]:
            conflict["duration"] += 1
            conflict["intensity"] *= 0.80  # ~2.6 yr half-life at 10-month tick
            if conflict["intensity"] < 0.05 or conflict["duration"] > 25:
                self.active_conflicts.remove(conflict)

        if len(self.nations) < 2:
            return

        # Assess each nation dyad
        for i, na in enumerate(self.nations):
            for j, nb in enumerate(self.nations):
                if i >= j:
                    continue

                # Skip if already in conflict
                already = any(
                    c for c in self.active_conflicts
                    if set(c["nations"]) == {na.id, nb.id}
                )
                if already:
                    continue

                prob = self.conflict_probability(na, nb, macro_state)

                if self.rng.random() < prob:
                    # New conflict
                    midpoint_lat = (na.center_lat + nb.center_lat) / 2
                    midpoint_lng = (na.center_lng + nb.center_lng) / 2
                    dist = self._great_circle_deg(
                        na.center_lat, na.center_lng,
                        nb.center_lat, nb.center_lng,
                    )

                    self.active_conflicts.append({
                        "nations": [na.id, nb.id],
                        "nation_names": [na.name, nb.name],
                        "lat": midpoint_lat,
                        "lng": midpoint_lng,
                        "radius": max(3.0, dist * 0.4),
                        "intensity": 0.3 + 0.4 * macro_state.social_tension,
                        "duration": 0,
                        "cause": "resource_competition" if macro_state.fossil_fuels < 0.5
                                 else "territorial",
                    })

                    # Worsen relations symmetrically. Use add_edge for both
                    # directions so a one-sided edge can never KeyError on the
                    # reverse write (edges are normally created in pairs by
                    # _update_relations; this stays robust if that ever changes).
                    if self.relation_graph.has_edge(na.id, nb.id):
                        w = self.relation_graph[na.id][nb.id]["weight"]
                        new_w = max(-1.0, w - 0.3)
                        self.relation_graph.add_edge(na.id, nb.id, weight=new_w)
                        self.relation_graph.add_edge(nb.id, na.id, weight=new_w)

    def conflict_probability(
        self,
        nation_a: NationState,
        nation_b: NationState,
        macro_state: 'MacroState',
    ) -> float:
        """
        Logistic conflict-probability model.

        Returns the per-tick probability of conflict initiation between this
        dyad, from a logistic model and then **capped at 0.10/tick** to bound the
        ignition rate. The calibration target (active-conflict prevalence in the
        5-nation BAU run, see the class-level CONFLICT_* block) is reached by
        accumulation across ticks, not by a single high per-tick value.

        Sources:
        - Hughes (2019), International Futures (IFs) conflict module
        - Russett (1993), Oneal & Russett (1999) — liberal peace
        - Bremer (1992) — Dangerous Dyads, parity effect
        - Homer-Dixon (1999) — environmental scarcity and conflict
        """
        if nation_a.population < 3 or nation_b.population < 3:
            return 0.0

        # Resource scarcity competition
        resource_scarcity = 1.0 - macro_state.fossil_fuels
        resource_comp = resource_scarcity * 0.5

        # Power parity (near-peer = more conflict)
        power_a = nation_a.total_wealth + nation_a.total_military * 10
        power_b = nation_b.total_wealth + nation_b.total_military * 10
        parity = float(np.clip(
            1.0 - abs(power_a - power_b) / (max(power_a, power_b) + 1), 0.0, 1.0))

        # Trade interdependence — liberal peace theory.
        # Russett (1993) Grasping the Democratic Peace, Princeton UP;
        # Oneal & Russett (1999) The Kantian peace, World Politics 52(1), 1-37.
        trade_vol = 0.0
        if self.trade_graph.has_edge(nation_a.id, nation_b.id):
            trade_vol = self.trade_graph[nation_a.id][nation_b.id].get("weight", 0)
        trade_interdep = min(1.0, trade_vol / 10.0)

        # Diplomatic history
        relation = 0.0
        if self.relation_graph.has_edge(nation_a.id, nation_b.id):
            relation = self.relation_graph[nation_a.id][nation_b.id].get("weight", 0)

        # Shared alliances
        shared = len(nation_a.alliances & nation_b.alliances)
        alliance_factor = min(1.0, shared / 3.0)

        # Territorial proximity
        dist = self._great_circle_deg(
            nation_a.center_lat, nation_a.center_lng,
            nation_b.center_lat, nation_b.center_lng,
        )
        territory_overlap = max(0, 1.0 - dist / 15.0)

        # Logistic model: P = sigmoid(sum of weighted factors)
        logit = (
            self.CONFLICT_BASE_RATE +
            self.CONFLICT_RESOURCE_COEFF * resource_comp +
            self.CONFLICT_PARITY_COEFF * parity +
            self.CONFLICT_TRADE_COEFF * trade_interdep +
            self.CONFLICT_TENSION_COEFF * macro_state.social_tension +
            self.CONFLICT_TERRITORY_COEFF * territory_overlap +
            self.CONFLICT_ALLIANCE_COEFF * alliance_factor +
            self.CONFLICT_HISTORY_COEFF * max(0, relation)
        )

        probability = 1.0 / (1.0 + np.exp(-logit))
        # Cap at 0.10/tick (see docstring): bounds the ignition rate; sustained
        # prevalence comes from accumulation across ticks, not one large value.
        return float(np.clip(probability, 0.0, 0.1))

    # ------------------------------------------------------------------
    # Negotiations
    # ------------------------------------------------------------------

    def _conduct_negotiations(self, macro_state: 'MacroState'):
        """Periodic climate and trade negotiations between nations."""
        if len(self.nations) < 2:
            return

        # Climate summit every ~24 macro ticks (~2 years).
        # FIX (v0.2): the previous code used `sum(n.age for n in self.nations) % 24`,
        # which fires every 24/N ticks (for N nations all aging at +1/tick) — i.e.,
        # every 5 ticks with 5 nations and every 24 ticks with 24 nations.
        # We track our own counter so the cadence is independent of nation count.
        self._negotiation_counter = getattr(self, "_negotiation_counter", 0) + 1
        if self._negotiation_counter % 24 == 0:
            # Each nation proposes a climate pledge based on policy
            for nation in self.nations:
                if nation.carbon_policy > 0:
                    nation.climate_pledge = min(1.0, nation.climate_pledge + 0.05)
                else:
                    nation.climate_pledge = max(0, nation.climate_pledge - 0.02)

            self.negotiation_history.append({
                "type": "climate_summit",
                "year": macro_state.year,
                "participants": len(self.nations),
                "avg_pledge": np.mean([n.climate_pledge for n in self.nations]),
            })

    # ------------------------------------------------------------------
    # Technology Diffusion
    # ------------------------------------------------------------------

    def _diffuse_technology(self):
        """
        Knowledge transfer between trading partners.

        FIX B2: trade_graph is a DiGraph and _resolve_trade writes both
        (a, b) and (b, a) edges with the same weight. Iterating edges()
        therefore visits each dyad twice; the original code credited the
        lower-tech nation on both visits, doubling the diffusion rate
        relative to the calibrated intent. The `na_id >= nb_id` filter
        below ensures each unordered dyad is processed exactly once.
        """
        for edge in self.trade_graph.edges(data=True):
            na_id, nb_id, data = edge
            if na_id >= nb_id:
                continue
            volume = data.get("weight", 0)
            if volume < 0.1:
                continue

            na = next((n for n in self.nations if n.id == na_id), None)
            nb = next((n for n in self.nations if n.id == nb_id), None)
            if na is None or nb is None:
                continue

            # Technology flows from high to low
            tech_diff = na.technology_level - nb.technology_level
            transfer_rate = 0.001 * volume * abs(tech_diff)

            if tech_diff > 0:
                nb.technology_level += transfer_rate
            else:
                na.technology_level += transfer_rate

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
