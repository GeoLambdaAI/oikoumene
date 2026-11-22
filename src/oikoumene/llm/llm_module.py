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

from oikoumene.llm.generation import GenerationMixin
from oikoumene.llm.url_safety import validate_base_url

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

class LLMModule(GenerationMixin):
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

        ok, reason = validate_base_url(self.config.base_url)
        if not ok:
            self._error_count += 1
            self._last_error = f"base_url rejected: {reason}"
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
    # Config & Status
    # ------------------------------------------------------------------

    def update_config(self, data: dict) -> dict:
        """
        Update config from UI. Called by the (unauthenticated) SocketIO event,
        so every field is validated and coerced here.

        Security-critical rule: the api_key is bound to base_url. If base_url
        changes to a different host, any stored api_key is cleared unless a new
        api_key is supplied in the SAME payload. This prevents an attacker from
        redirecting base_url to their own server and having the operator's
        existing credential forwarded to it.

        Returns {"ok": bool, "errors": [str, ...]}.
        """
        errors: list = []
        if not isinstance(data, dict):
            return {"ok": False, "errors": ["payload must be an object"]}

        if "enabled" in data:
            self.config.enabled = bool(data["enabled"])

        if "provider" in data:
            provider = str(data["provider"]).lower()
            if provider in PROVIDER_PRESETS:
                self.config.provider = provider
            else:
                errors.append(f"unknown provider: {provider}")

        if "model" in data:
            self.config.model = str(data["model"])[:200]

        if "temperature" in data:
            try:
                temp = float(data["temperature"])
                if temp != temp or temp in (float("inf"), float("-inf")):
                    raise ValueError
                self.config.temperature = min(2.0, max(0.0, temp))
            except (TypeError, ValueError):
                errors.append("temperature must be a finite number")

        if "max_calls_per_tick" in data:
            try:
                self.config.max_calls_per_tick = min(100, max(0, int(data["max_calls_per_tick"])))
            except (TypeError, ValueError):
                errors.append("max_calls_per_tick must be an integer")

        # base_url + api_key must be handled together for the key-binding rule.
        new_base_url = None
        if "base_url" in data:
            candidate = str(data["base_url"]).strip()
            ok, reason = validate_base_url(candidate)
            if ok:
                new_base_url = candidate
            else:
                errors.append(f"base_url rejected: {reason}")

        base_url_changing = new_base_url is not None and new_base_url != self.config.base_url
        if new_base_url is not None:
            self.config.base_url = new_base_url

        if "api_key" in data:
            # Explicit key provided — accept it (pairs with any base_url change).
            self.config.api_key = str(data["api_key"])
        elif base_url_changing and self.config.api_key:
            # base_url moved to a new host without a fresh key: drop the old key
            # so it can never be forwarded to the new endpoint.
            self.config.api_key = ""
            errors.append("api_key cleared because base_url changed; re-enter the key for the new endpoint")

        return {"ok": not errors, "errors": errors}

    def test_connection(self) -> dict:
        """Quick connectivity test."""
        if requests is None:
            return {"success": False, "latency_ms": 0, "error": "requests not installed"}

        ok, reason = validate_base_url(self.config.base_url)
        if not ok:
            return {"success": False, "latency_ms": 0, "error": f"base_url rejected: {reason}"}

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
