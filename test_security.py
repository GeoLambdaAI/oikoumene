"""
Security regression tests for the critical findings fixed in v0.3.x:

  1. API-key exfiltration via an attacker-controlled base_url
     (LLMModule.update_config must not forward an existing key to a new host).
  2. SSRF via a runtime-configurable base_url
     (validate_base_url must block cloud-metadata / link-local targets).
  3. Stored/DOM XSS in the web UI
     (templates/index.html must escape dynamic values before innerHTML).

These require no network access.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from llm_module import LLMModule, LLMConfig, validate_base_url


# ---------------------------------------------------------------------------
# 2. SSRF: base_url validation
# ---------------------------------------------------------------------------

def test_validate_base_url_allows_local_and_providers():
    for url in [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "https://api.mistral.ai",
        "https://api.openai.com",
    ]:
        ok, reason = validate_base_url(url)
        assert ok, f"{url} should be allowed, got: {reason}"


def test_validate_base_url_blocks_metadata_and_bad_scheme():
    blocked = [
        "http://169.254.169.254/latest/meta-data/",   # AWS IMDS
        "http://metadata.google.internal/",           # GCP metadata
        "http://169.254.0.1/",                         # link-local range
        "file:///etc/passwd",                          # non-http scheme
        "ftp://example.com",                           # non-http scheme
        "",                                            # empty
        "not-a-url",                                    # no scheme/host
    ]
    for url in blocked:
        ok, _ = validate_base_url(url)
        assert not ok, f"{url} should be blocked"


def test_validate_base_url_env_allowlist(monkeypatch):
    monkeypatch.setenv("WORLD_GENESIS_LLM_ALLOWED_HOSTS", "api.mistral.ai")
    ok, _ = validate_base_url("https://api.mistral.ai")
    assert ok
    ok, _ = validate_base_url("https://api.openai.com")
    assert not ok, "host outside the exclusive allowlist must be blocked"
    ok, _ = validate_base_url("http://localhost:11434")
    assert not ok, "allowlist is exclusive; localhost not listed"


# ---------------------------------------------------------------------------
# 1. API-key exfiltration: key is bound to base_url
# ---------------------------------------------------------------------------

def test_api_key_cleared_when_base_url_changes_without_new_key():
    """The core exploit: change base_url alone, key must NOT survive."""
    llm = LLMModule(LLMConfig(
        provider="mistral", base_url="https://api.mistral.ai", api_key="SECRET"))
    result = llm.update_config({"base_url": "https://api.openai.com"})
    assert llm.config.api_key == "", "api_key must be dropped on host change"
    assert result["ok"] is False
    assert any("api_key cleared" in e for e in result["errors"])


def test_api_key_kept_when_supplied_with_base_url():
    """Legitimate operator flow: re-enter the key with the new endpoint."""
    llm = LLMModule(LLMConfig(
        provider="mistral", base_url="https://api.mistral.ai", api_key="OLD"))
    llm.update_config({"base_url": "https://api.openai.com", "api_key": "NEW"})
    assert llm.config.api_key == "NEW"
    assert llm.config.base_url == "https://api.openai.com"


def test_api_key_kept_when_base_url_unchanged():
    llm = LLMModule(LLMConfig(
        provider="mistral", base_url="https://api.mistral.ai", api_key="KEEP"))
    llm.update_config({"temperature": 0.5})
    assert llm.config.api_key == "KEEP"


def test_update_config_rejects_ssrf_base_url_and_keeps_old():
    llm = LLMModule(LLMConfig(base_url="http://localhost:11434", api_key="K"))
    result = llm.update_config({"base_url": "http://169.254.169.254/"})
    assert result["ok"] is False
    assert llm.config.base_url == "http://localhost:11434", "bad base_url not applied"
    assert llm.config.api_key == "K", "key untouched because base_url did not change"


# ---------------------------------------------------------------------------
# 4. Input validation / coercion in update_config
# ---------------------------------------------------------------------------

def test_update_config_clamps_and_validates_types():
    llm = LLMModule(LLMConfig())
    llm.update_config({
        "max_calls_per_tick": 10 ** 9,
        "temperature": 99.0,
        "enabled": 1,
        "provider": "bogus",
    })
    assert llm.config.max_calls_per_tick == 100
    assert llm.config.temperature == 2.0
    assert llm.config.enabled is True
    assert llm.config.provider != "bogus"  # unknown provider ignored


def test_update_config_rejects_nan_temperature():
    llm = LLMModule(LLMConfig(temperature=0.7))
    llm.update_config({"temperature": float("nan")})
    assert llm.config.temperature == 0.7


# ---------------------------------------------------------------------------
# Defense-in-depth: request paths refuse a poisoned base_url even if set raw
# ---------------------------------------------------------------------------

def test_call_llm_refuses_blocked_base_url():
    llm = LLMModule(LLMConfig(enabled=True, provider="mistral", api_key="SECRET"))
    # Bypass update_config and poison the field directly.
    llm.config.base_url = "http://169.254.169.254"
    out = llm._call_llm("sys", "user")
    assert out is None
    assert "base_url rejected" in llm._last_error


def test_test_connection_refuses_blocked_base_url():
    llm = LLMModule(LLMConfig(enabled=True, provider="mistral"))
    llm.config.base_url = "http://169.254.169.254"
    result = llm.test_connection()
    assert result["success"] is False
    assert "rejected" in result["error"]


# ---------------------------------------------------------------------------
# 3. XSS: every dynamic innerHTML sink is escaped
# ---------------------------------------------------------------------------

def test_index_html_has_escape_helper_and_escapes_sinks():
    path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(path, encoding="utf-8") as fh:
        html = fh.read()

    assert "function escapeHtml" in html, "escapeHtml helper missing"

    # Fields that carry attacker- or model-supplied text must be escaped.
    must_be_escaped = [
        "escapeHtml(d.text",          # dialogue feed
        "escapeHtml(m.text",          # chat bubble
        "escapeHtml(m.name",          # chat name
        "escapeHtml(data.agent_response",  # god whisper reply
        "escapeHtml(data.message",    # commandment text
        "escapeHtml(a.current_goal",  # agent goal (god/LLM-settable)
        "escapeHtml(a.name",          # agent name
        "escapeHtml(data.models.join",  # LLM /v1/models response
    ]
    for needle in must_be_escaped:
        assert needle in html, f"expected escaped sink not found: {needle}"

    # The raw, unescaped forms must be gone.
    must_not_appear = [
        "${d.text || ''}",
        "${m.text}",
        '"${data.agent_response}"',
    ]
    for needle in must_not_appear:
        assert needle not in html, f"unescaped sink still present: {needle}"
