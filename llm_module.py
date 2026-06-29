"""
LLM Social Cognition Module — Theory-of-Mind for autonomous agents.

Provides language-based social reasoning that materially affects simulation
outcomes. When enabled, social actions (trade, govern, socialize, teach,
alliance) are mediated by LLM dialogue, modifying numerical outcomes.

When disabled, returns structured fallback responses and all outcomes
revert to the existing trait-based calculations. Zero simulation impact.

Supports:
- Local models via Ollama HTTP API (Qwen 3.5, Mistral, etc.)
- Any OpenAI-compatible API endpoint
- Disabled mode with deterministic fallback

Architecture: Kahneman's Dual Process Theory
- System 1 (JEPA): fast, every tick, decides WHAT to do
- System 2 (LLM): slow, social actions only, decides HOW WELL it's done

Reference: LeCun (2022) "A Path Towards Autonomous Machine Intelligence"
Section 3.3: Theory of Mind for social agents.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np

try:
    import requests
except ImportError:
    requests = None  # type: ignore


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LLMConfig:
    """
    LLM configuration.

    Privacy-first defaults: agent cognition runs on a LOCAL model (Ollama, no
    data leaves the machine) or EU-based Mistral AI (GDPR-friendly). Any
    OpenAI-compatible endpoint also works (set provider/base_url/model, e.g.
    OpenAI) — note that a non-EU provider receives agent context off-machine,
    which the operator is responsible for assessing under EU AI Act / GDPR.
    """
    enabled: bool = False
    provider: str = "ollama"                   # "ollama" | "mistral" | "openai" (OpenAI-compatible)
    base_url: str = "http://localhost:11434"    # Ollama default
    model: str = "qwen3:8b"                    # Ollama default model
    api_key: str = ""                          # Required for Mistral AI
    max_calls_per_tick: int = 5
    max_tokens: int = 150
    temperature: float = 0.7
    timeout_seconds: float = 10.0
    trigger_actions: list = field(default_factory=lambda: [
        "socialize", "trade", "govern", "teach", "form_alliance", "negotiate"
    ])

# Provider presets. Defaults are privacy-first (local Ollama / EU Mistral);
# openai is an opt-in OpenAI-compatible endpoint (data leaves the EU — see LLMConfig).
PROVIDER_PRESETS = {
    "ollama": {"base_url": "http://localhost:11434", "model": "qwen3:8b"},
    "mistral": {"base_url": "https://api.mistral.ai", "model": "mistral-small-latest"},
    "openai": {"base_url": "https://api.openai.com", "model": "gpt-4o-mini"},
}


# ============================================================================
# LLM Module
# ============================================================================

class LLMModule:
    """
    Social cognition engine for autonomous agents.

    When enabled: social actions are mediated by LLM dialogue. The LLM
    generates proposals, reactions, and evaluations that produce numerical
    modifiers (0.3-2.5x) applied to base trait-only calculations.

    When disabled: all methods return fallback responses with modifier=1.0,
    producing identical simulation output to the pre-LLM codebase.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._tick_call_count: int = 0
        self._total_calls: int = 0
        self._error_count: int = 0
        self._avg_latency: float = 0.0
        self._last_error: str = ""

    def reset_tick_counter(self):
        """Call at start of each World.step()."""
        self._tick_call_count = 0

    def can_call(self, is_user_chat: bool = False) -> bool:
        """
        Check if we're allowed to call the LLM.

        is_user_chat: if True, bypass tick rate limit (user-initiated chat
        is not part of the simulation tick budget).
        """
        if not self.config.enabled or requests is None:
            return False
        if is_user_chat:
            return True  # User chat always allowed when LLM is enabled
        return self._tick_call_count < self.config.max_calls_per_tick

    # ------------------------------------------------------------------
    # Social Action Methods
    # ------------------------------------------------------------------

    def generate_trade_negotiation(
        self, agent, partner, current_trade_value: float, context: dict
    ) -> dict:
        """
        Trade negotiation between two agents.

        Returns {"modifier": float, "proposer_text": str, "responder_text": str,
                 "accepted": bool, "negotiated_value": float}

        modifier: multiplier on base trade_value (0.3-2.5)
        When disabled: modifier=1.0
        """
        if not self.can_call():
            return self._fallback_trade(agent, partner, current_trade_value)

        system = self._build_system_prompt(agent)
        user = (
            f"You are negotiating a trade with {partner.name} "
            f"(wealth: {partner.wealth:.0f}, traits: cooperation={partner.traits['cooperation']:.2f}). "
            f"Your relationship: {agent.relationships.get(partner.id, 0):.2f}. "
            f"Base trade value: {current_trade_value:.1f}. "
            f"Make a proposal in 1-2 sentences, then provide JSON: "
            f'{{\"modifier\": <0.3-2.5>, \"accepted\": true/false}}'
        )

        raw = self._call_llm(system, user)
        if raw is None:
            return self._fallback_trade(agent, partner, current_trade_value)

        parsed = self._parse_structured_response(raw, ["modifier"])
        modifier = float(np.clip(parsed.get("modifier", 1.0), 0.3, 2.5))

        return {
            "modifier": modifier,
            "proposer_text": raw[:200],
            "responder_text": "",
            "accepted": parsed.get("accepted", True),
            "negotiated_value": current_trade_value * modifier,
        }

    def generate_governance_speech(
        self, agent, audience_agents: list, context: dict
    ) -> dict:
        """
        Governance speech and audience evaluation.

        Returns {"influence_modifier": float, "speech_text": str,
                 "persuasiveness": float}

        influence_modifier: multiplier on base influence (0.3-2.0)
        When disabled: influence_modifier=1.0
        """
        if not self.can_call():
            return self._fallback_govern(agent)

        system = self._build_system_prompt(agent)
        n_audience = len(audience_agents)
        tension = context.get("social_tension", 0.25)
        user = (
            f"You are addressing {n_audience} people in your settlement. "
            f"Social tension is {tension:.0%}. "
            f"Give a leadership speech in 1-3 sentences, then provide JSON: "
            f'{{\"influence_modifier\": <0.3-2.0>, \"persuasiveness\": <0.0-1.0>}}'
        )

        raw = self._call_llm(system, user)
        if raw is None:
            return self._fallback_govern(agent)

        parsed = self._parse_structured_response(raw, ["influence_modifier"])
        return {
            "influence_modifier": float(np.clip(parsed.get("influence_modifier", 1.0), 0.3, 2.0)),
            "speech_text": raw[:200],
            "persuasiveness": float(np.clip(parsed.get("persuasiveness", 0.5), 0.0, 1.0)),
        }

    def generate_social_dialogue(
        self, agent, partner, compatibility: float, context: dict
    ) -> dict:
        """
        Social conversation between agents.

        Returns {"relationship_modifier": float, "happiness_modifier": float,
                 "agent_text": str, "partner_text": str, "connection_quality": float}

        When disabled: both modifiers = 1.0
        """
        if not self.can_call():
            return self._fallback_socialize(agent, partner, compatibility)

        system = self._build_system_prompt(agent)
        user = (
            f"You meet {partner.name} (compatibility: {compatibility:.0%}). "
            f"Have a brief conversation in 1-2 sentences, then provide JSON: "
            f'{{\"relationship_modifier\": <0.5-2.0>, \"happiness_modifier\": <0.5-2.0>, '
            f'"connection_quality": <0.0-1.0>}}'
        )

        raw = self._call_llm(system, user)
        if raw is None:
            return self._fallback_socialize(agent, partner, compatibility)

        parsed = self._parse_structured_response(raw, ["relationship_modifier"])
        return {
            "relationship_modifier": float(np.clip(parsed.get("relationship_modifier", 1.0), 0.5, 2.0)),
            "happiness_modifier": float(np.clip(parsed.get("happiness_modifier", 1.0), 0.5, 2.0)),
            "agent_text": raw[:200],
            "partner_text": "",
            "connection_quality": float(np.clip(parsed.get("connection_quality", 0.5), 0.0, 1.0)),
        }

    def generate_teaching(
        self, teacher, student, topic: str, context: dict
    ) -> dict:
        """
        Teaching interaction.

        Returns {"skill_transfer_modifier": float, "lesson_text": str,
                 "clarity_score": float}

        When disabled: skill_transfer_modifier=1.0
        """
        if not self.can_call():
            return self._fallback_teach(teacher, topic)

        system = self._build_system_prompt(teacher)
        user = (
            f"Teach {student.name} about {topic}. "
            f"Student's current skill: {student.skills.get_level(topic):.2f}. "
            f"Explain in 1-2 sentences, then JSON: "
            f'{{\"skill_transfer_modifier\": <0.5-2.0>, \"clarity_score\": <0.0-1.0>}}'
        )

        raw = self._call_llm(system, user)
        if raw is None:
            return self._fallback_teach(teacher, topic)

        parsed = self._parse_structured_response(raw, ["skill_transfer_modifier"])
        return {
            "skill_transfer_modifier": float(np.clip(parsed.get("skill_transfer_modifier", 1.0), 0.5, 2.0)),
            "lesson_text": raw[:200],
            "clarity_score": float(np.clip(parsed.get("clarity_score", 0.5), 0.0, 1.0)),
        }

    def generate_god_response(
        self, agent, divine_message: str, msg_type: str
    ) -> dict:
        """
        Agent's reaction to a God Mode message.

        Personality determines reaction:
        - High intelligence: questions the message
        - High cooperation: accepts obediently
        - High risk_tolerance: acts boldly

        Returns {"reaction_text": str, "compliance": float}
        """
        if not self.can_call():
            compliance = (
                agent.traits["cooperation"] * 0.5 +
                (1.0 - agent.traits["intelligence"] * 0.3) +
                getattr(agent, 'divine_trust', 0.5) * 0.2
            )
            return {"reaction_text": "", "compliance": float(np.clip(compliance, 0.1, 0.95))}

        system = self._build_system_prompt(agent)
        user = (
            f'A divine voice says: "{divine_message}"\n'
            f"React in 1 sentence based on your personality. "
            f"Then JSON: {{\"compliance\": <0.0-1.0>}}"
        )

        raw = self._call_llm(system, user)
        if raw is None:
            return {"reaction_text": "", "compliance": 0.5}

        parsed = self._parse_structured_response(raw, ["compliance"])
        return {
            "reaction_text": raw[:200],
            "compliance": float(np.clip(parsed.get("compliance", 0.5), 0.0, 1.0)),
        }

    def _generate_trait_response(self, agent, user_message: str,
                                 context: str = "as_peer") -> dict:
        """
        Generate a personality-driven response WITHOUT LLM.
        Uses agent traits, current state, and message keywords to create
        plausible dialogue. Always produces an interesting response.
        """
        name = agent.name
        energy = agent.energy
        wealth = agent.wealth
        action = getattr(agent, 'current_action', 'idle')
        goal = getattr(agent, 'current_goal', 'survive')

        # Dominant trait determines personality
        traits = agent.traits
        dominant = max(traits, key=traits.get)
        msg_lower = user_message.lower()

        # Tone from traits
        if traits.get("cooperation", 0.5) > 0.7:
            tone = "friendly"
        elif traits.get("risk_tolerance", 0.5) > 0.7:
            tone = "hostile"
        elif traits.get("curiosity", 0.5) > 0.7:
            tone = "curious"
        elif energy < 30:
            tone = "fearful"
        else:
            tone = "neutral"

        # Build response based on personality + situation + keywords
        responses = []

        # Greeting / introduction
        if any(w in msg_lower for w in ["hello", "hi", "hey", "greet", "who"]):
            if traits.get("sociability", 0.5) > 0.6:
                responses.append(f"Welcome, stranger! I am {name}. Always glad to meet new faces.")
            elif traits.get("intelligence", 0.5) > 0.6:
                responses.append(f"I am {name}. State your purpose clearly.")
            elif traits.get("ambition", 0.5) > 0.7:
                responses.append(f"{name} here. I'm busy building something great. What do you need?")
            else:
                responses.append(f"I'm {name}. What brings you here?")

        # Questions about state
        elif any(w in msg_lower for w in ["how are", "how do you", "feeling", "okay"]):
            if energy < 30:
                responses.append(f"Honestly? I'm struggling. Energy is low ({energy:.0f}), and food is scarce around here.")
            elif wealth > 50:
                responses.append(f"Doing well, actually! Business is good (wealth: {wealth:.0f}). Life could be worse.")
            elif traits.get("resilience", 0.5) > 0.7:
                responses.append(f"I endure. That's what I do. Currently focused on {goal}.")
            else:
                responses.append(f"Getting by. I'm trying to {goal} right now.")

        # Food / resources
        elif any(w in msg_lower for w in ["food", "eat", "hungry", "resource"]):
            if energy < 40:
                responses.append(f"Food is exactly what I need! My energy is only {energy:.0f}. Do you know where to find some?")
            else:
                responses.append(f"I have enough for now. The land here provides, though it's getting harder each season.")

        # Trade / wealth
        elif any(w in msg_lower for w in ["trade", "buy", "sell", "wealth", "money"]):
            if traits.get("ambition", 0.5) > 0.6:
                responses.append(f"Now you're speaking my language! I have {wealth:.0f} wealth. What do you offer?")
            else:
                responses.append(f"I'm not much of a trader. I prefer simpler things.")

        # Exploration / travel
        elif any(w in msg_lower for w in ["explore", "travel", "move", "north", "south", "east", "west", "go"]):
            if traits.get("curiosity", 0.5) > 0.6:
                responses.append(f"I love exploring! I've been wanting to see what lies beyond these lands.")
                return {"text": responses[0], "tone": "curious", "goal_change": "explore"}
            else:
                responses.append(f"Travel? Too risky for my taste. I'd rather stay where I know the terrain.")

        # Danger / conflict
        elif any(w in msg_lower for w in ["danger", "war", "fight", "attack", "conflict", "flee"]):
            if traits.get("risk_tolerance", 0.5) > 0.7:
                responses.append(f"Danger? I laugh at danger. Let them come!")
            elif traits.get("cooperation", 0.5) > 0.7:
                responses.append(f"We should work together for safety. Alone we fall, together we stand.")
            else:
                responses.append(f"That's... concerning. Maybe I should think about moving somewhere safer.")
                return {"text": responses[0], "tone": "fearful", "goal_change": "migrate"}

        # Default: personality-driven
        else:
            personality_responses = {
                "intelligence": f"Hmm, interesting point. Let me think about that... My analysis suggests we need to focus on {goal}.",
                "creativity": f"Oh! That gives me an idea. What if we approached {goal} from a completely different angle?",
                "sociability": f"That's a great conversation topic! You know, I was just telling my neighbor about something similar.",
                "ambition": f"I appreciate the thought, but I'm focused on bigger things. My current goal: {goal}.",
                "risk_tolerance": f"Ha! I've heard bolder claims. But sure, I'm always up for something new.",
                "cooperation": f"I think we can find common ground here. Let's work on this together.",
                "resilience": f"I've seen worse times. Whatever happens, I'll adapt. Right now I'm working on {goal}.",
                "curiosity": f"Tell me more! I want to understand everything about this. What else do you know?",
            }
            responses.append(personality_responses.get(dominant,
                f"I hear you. I'm currently focused on {goal}, but I'm listening."))

        text = responses[0] if responses else f"{name} considers your words carefully."
        return {"text": text, "tone": tone, "goal_change": None}

    def generate_direct_chat(self, agent, user_message: str,
                             context: str = "as_peer",
                             world_state: dict = None) -> dict:
        """
        Direct conversation between user and an agent.

        Works with OR without LLM:
        - With LLM: full language model response
        - Without LLM: rich trait-based dialogue system

        Returns {"text": str, "tone": str, "goal_change": str|None, "llm_used": bool}
        """
        if not self.can_call(is_user_chat=True):
            result = self._generate_trait_response(agent, user_message, context)
            result["llm_used"] = False
            return result

        system = self._build_system_prompt(agent)
        # Enrich with world context
        extra = ""
        if world_state:
            extra = (f" Local food: {world_state.get('local_food', 0):.0f}, "
                     f"nearby agents: {world_state.get('nearby_agents', 0)}. ")

        role_map = {
            "as_god": "A divine being addresses you. ",
            "as_trader": "A traveling merchant speaks to you. ",
            "as_peer": "A fellow person approaches you. ",
        }
        role = role_map.get(context, "Someone speaks to you. ")

        user_prompt = (
            f"{role}{extra}\n"
            f'They say: "{user_message}"\n\n'
            f"Respond in character (1-3 sentences). Then JSON: "
            f'{{\"tone\": \"friendly\"|\"hostile\"|\"curious\"|\"fearful\"|\"neutral\", '
            f'"goal_change\": null or \"eat\"|\"explore\"|\"trade\"|\"migrate\"}}'
        )

        raw = self._call_llm(system, user_prompt)
        if raw is None:
            # LLM call failed — fall back to traits
            result = self._generate_trait_response(agent, user_message, context)
            result["llm_used"] = False
            result["error"] = self._last_error
            return result

        parsed = self._parse_structured_response(raw, ["tone"])
        # Extract text before JSON block
        text = raw.split("{")[0].strip() if "{" in raw else raw.strip()

        return {
            "text": text[:300],
            "tone": parsed.get("tone", "neutral"),
            "goal_change": parsed.get("goal_change"),
            "llm_used": True,
        }

    def generate_group_chat(self, agents: list, user_message: str,
                            context: str = "as_peer") -> list[dict]:
        """
        Send a message to multiple agents. Each responds individually.
        Works with OR without LLM (trait-based fallback for each agent).
        Returns list of {"agent_id", "agent_name", "text", "tone", "llm_used"}.
        """
        results = []
        for agent in agents:
            resp = self.generate_direct_chat(agent, user_message, context)
            results.append({
                "agent_id": agent.id,
                "agent_name": agent.name,
                "text": resp.get("text", ""),
                "tone": resp.get("tone", "neutral"),
                "llm_used": resp.get("llm_used", False),
            })
        return results

    # ------------------------------------------------------------------
    # System Prompt Builder
    # ------------------------------------------------------------------

    def _build_system_prompt(self, agent) -> str:
        """Build personality-rich system prompt from agent state."""
        top_traits = sorted(agent.traits.items(), key=lambda x: x[1], reverse=True)[:3]
        trait_str = ", ".join(f"{k}: {v:.2f}" for k, v in top_traits)

        # Speaking style from dominant trait
        dominant = top_traits[0][0]
        style_map = {
            "ambition": "You speak confidently and assertively.",
            "cooperation": "You speak warmly and seek consensus.",
            "creativity": "You speak imaginatively with unusual ideas.",
            "intelligence": "You speak analytically and precisely.",
            "sociability": "You speak enthusiastically and engagingly.",
            "risk_tolerance": "You speak boldly and make daring proposals.",
            "resilience": "You speak steadily with calm determination.",
            "curiosity": "You speak inquisitively, always asking questions.",
        }
        style = style_map.get(dominant, "You speak plainly.")

        return (
            f"You are {agent.name}, age {agent.age}, generation {agent.generation}. "
            f"Top traits: {trait_str}. "
            f"Energy: {agent.energy:.0f}/100, wealth: {agent.wealth:.0f}, "
            f"happiness: {agent.happiness:.0f}/100. "
            f"{style} "
            f"Respond in 1-3 sentences MAX. Then provide a JSON block with numerical fields."
        )

    # ------------------------------------------------------------------
    # LLM HTTP Client
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str) -> Optional[str]:
        """
        Make HTTP call to configured LLM endpoint.

        Supports:
        - Mistral AI (https://api.mistral.ai/v1/chat/completions)
        - Ollama local (/api/generate)
        - Any OpenAI-compatible endpoint (/v1/chat/completions)
        """
        if requests is None:
            return None

        self._tick_call_count += 1
        self._total_calls += 1
        t0 = time.time()

        try:
            if self.config.provider == "ollama":
                resp = requests.post(
                    f"{self.config.base_url}/api/generate",
                    json={
                        "model": self.config.model,
                        "system": system,
                        "prompt": user,
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens,
                        },
                    },
                    timeout=self.config.timeout_seconds,
                )
                resp.raise_for_status()
                result = resp.json().get("response", "")
            else:
                # Mistral AI and OpenAI-compatible use same chat/completions format
                headers = {"Content-Type": "application/json"}
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"

                resp = requests.post(
                    f"{self.config.base_url}/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": self.config.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                    },
                    timeout=self.config.timeout_seconds,
                )
                resp.raise_for_status()
                result = resp.json()["choices"][0]["message"]["content"]

            latency = (time.time() - t0) * 1000
            self._avg_latency = 0.9 * self._avg_latency + 0.1 * latency
            return result

        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)[:200]
            return None

    # ------------------------------------------------------------------
    # Response Parsing
    # ------------------------------------------------------------------

    def _parse_structured_response(self, raw_text: str, expected_fields: list) -> dict:
        """
        Extract JSON from LLM response text.

        LLM responses often contain text + JSON. We search for the last
        {...} block, parse it, and validate expected fields.
        If parsing fails, return default values.
        """
        # Find all JSON-like blocks
        matches = re.findall(r'\{[^{}]+\}', raw_text)
        if not matches:
            return {f: 1.0 for f in expected_fields}

        # Try the last match first (usually the structured output)
        for candidate in reversed(matches):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        return {f: 1.0 for f in expected_fields}

    # ------------------------------------------------------------------
    # Fallback Responses (deterministic, modifier=1.0)
    # ------------------------------------------------------------------

    def _fallback_trade(self, agent, partner, value: float) -> dict:
        templates = {
            "high_ambition": f"{agent.name} drives a hard bargain with {partner.name}.",
            "high_cooperation": f"{agent.name} proposes a fair exchange with {partner.name}.",
            "default": f"{agent.name} trades with {partner.name}.",
        }
        key = "high_ambition" if agent.traits["ambition"] > 0.7 else \
              "high_cooperation" if agent.traits["cooperation"] > 0.7 else "default"
        return {
            "modifier": 1.0, "proposer_text": templates[key],
            "responder_text": "", "accepted": True,
            "negotiated_value": value,
        }

    def _fallback_govern(self, agent) -> dict:
        if agent.traits["ambition"] > 0.7:
            text = f"{agent.name} commands attention with an authoritative address."
        elif agent.traits["cooperation"] > 0.7:
            text = f"{agent.name} speaks of unity and shared purpose."
        else:
            text = f"{agent.name} addresses the settlement."
        return {"influence_modifier": 1.0, "speech_text": text, "persuasiveness": 0.5}

    def _fallback_socialize(self, agent, partner, compat: float) -> dict:
        if compat > 0.7:
            text = f"{agent.name} and {partner.name} share a warm conversation."
        elif compat < 0.3:
            text = f"{agent.name} and {partner.name} have an awkward exchange."
        else:
            text = f"{agent.name} chats with {partner.name}."
        return {
            "relationship_modifier": 1.0, "happiness_modifier": 1.0,
            "agent_text": text, "partner_text": "",
            "connection_quality": compat,
        }

    def _fallback_teach(self, teacher, topic: str) -> dict:
        return {
            "skill_transfer_modifier": 1.0,
            "lesson_text": f"{teacher.name} explains {topic}.",
            "clarity_score": 0.5,
        }

    # ------------------------------------------------------------------
    # Config & Status
    # ------------------------------------------------------------------

    def update_config(self, data: dict):
        """Update config from UI. Called by SocketIO event."""
        for key in ["enabled", "provider", "base_url", "model",
                     "api_key", "temperature", "max_calls_per_tick"]:
            if key in data:
                setattr(self.config, key, data[key])

    def test_connection(self) -> dict:
        """Quick connectivity test."""
        if requests is None:
            return {"success": False, "latency_ms": 0, "error": "requests not installed"}

        t0 = time.time()
        try:
            if self.config.provider == "ollama":
                resp = requests.get(
                    f"{self.config.base_url}/api/tags",
                    timeout=5.0,
                )
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                latency = (time.time() - t0) * 1000
                return {"success": True, "latency_ms": round(latency, 1),
                        "models": models, "error": ""}
            else:
                # Mistral AI and OpenAI-compatible both support /v1/models
                headers = {}
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                resp = requests.get(
                    f"{self.config.base_url}/v1/models",
                    headers=headers,
                    timeout=5.0,
                )
                resp.raise_for_status()
                data = resp.json()
                models = [m.get("id", "?") for m in data.get("data", [])]
                latency = (time.time() - t0) * 1000
                return {"success": True, "latency_ms": round(latency, 1),
                        "models": models, "error": ""}
        except Exception as e:
            return {"success": False, "latency_ms": 0, "error": str(e)[:200]}

    def get_status(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "provider": self.config.provider,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "calls_this_tick": self._tick_call_count,
            "total_calls": self._total_calls,
            "avg_latency_ms": round(self._avg_latency, 1),
            "errors": self._error_count,
            "last_error": self._last_error,
        }
