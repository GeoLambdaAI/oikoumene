"""
Interactions mixin for GeopoliticalSystem (geopolitics.py): diplomatic relations,
trade, conflict assessment and conflict_probability, negotiations, technology diffusion.

Expects on the host class: the attributes nations, relation_graph, trade_graph,
active_conflicts, negotiation_history, _negotiation_counter, _conflict_assessments and
rng; the class constants CONFLICT_* (11) and _MAX_NEGOTIATION_HISTORY; and the
static method _great_circle_deg.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oikoumene.macro.macro import MacroState
    from oikoumene.society.geopolitics import NationState


class InteractionsMixin:
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

        self._conflict_assessments += 1
        if len(self.nations) < 2:
            return

        # --- Geographic gating -------------------------------------------------
        # Collect only proximate dyads; distant pairs cannot ignite (see class
        # constant docstring). O(N²) distance scan, but each check is cheap.
        eligible: list[tuple] = []
        nations = self.nations
        n = len(nations)
        for i in range(n):
            na = nations[i]
            for j in range(i + 1, n):
                nb = nations[j]
                dist = self._great_circle_deg(
                    na.center_lat, na.center_lng, nb.center_lat, nb.center_lng)
                if dist <= self.CONFLICT_MAX_DYAD_DEG:
                    eligible.append((na, nb, dist))

        # --- N-aware normalization + spin-up ----------------------------------
        n_elig = len(eligible)
        n_scale = min(1.0, self.CONFLICT_DYAD_REFERENCE / n_elig) if n_elig else 1.0
        spinup = min(1.0, self._conflict_assessments / self.CONFLICT_SPINUP_ASSESSMENTS)
        onset_scale = n_scale * spinup

        # Assess each eligible dyad
        for na, nb, dist in eligible:
            # Skip if already in conflict
            already = any(
                c for c in self.active_conflicts
                if set(c["nations"]) == {na.id, nb.id}
            )
            if already:
                continue

            prob = self.conflict_probability(na, nb, macro_state) * onset_scale

            if self.rng.random() < prob:
                # New conflict
                midpoint_lat = (na.center_lat + nb.center_lat) / 2
                midpoint_lng = (na.center_lng + nb.center_lng) / 2

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
            # Bound the history so it does not grow without limit over a long run.
            if len(self.negotiation_history) > self._MAX_NEGOTIATION_HISTORY:
                self.negotiation_history = (
                    self.negotiation_history[-self._MAX_NEGOTIATION_HISTORY:]
                )

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
