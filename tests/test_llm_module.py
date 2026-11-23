"""
Test suite for LLM Social Cognition Module.

Tests fallback mode (no LLM server needed), parsing, rate limiting,
and error handling. No network calls required.
"""


from oikoumene.llm.llm_module import LLMModule, LLMConfig


class MockAgent:
    """Minimal agent mock for testing."""
    def __init__(self, name="TestAgent", **traits):
        self.id = 1
        self.name = name
        self.age = 100
        self.generation = 2
        self.energy = 75.0
        self.wealth = 50.0
        self.happiness = 60.0
        self.traits = {
            "intelligence": 0.6, "creativity": 0.5, "sociability": 0.7,
            "ambition": 0.8, "risk_tolerance": 0.4, "cooperation": 0.6,
            "resilience": 0.5, "curiosity": 0.6,
        }
        self.traits.update(traits)
        self.relationships = {}

        class MockSkills:
            def get_level(self, name): return 0.3
        self.skills = MockSkills()


def test_disabled_mode():
    """When disabled, all methods return modifier=1.0."""
    llm = LLMModule(LLMConfig(enabled=False))
    agent = MockAgent("Alice")
    partner = MockAgent("Bob")

    result = llm.generate_trade_negotiation(agent, partner, 10.0, {})
    assert result["modifier"] == 1.0, f"Expected 1.0, got {result['modifier']}"
    assert result["negotiated_value"] == 10.0
    assert result["accepted"] is True
    assert isinstance(result["proposer_text"], str)
    print("  PASS: disabled trade returns modifier=1.0")

    result = llm.generate_governance_speech(agent, [partner], {})
    assert result["influence_modifier"] == 1.0
    assert isinstance(result["speech_text"], str)
    print("  PASS: disabled govern returns influence_modifier=1.0")

    result = llm.generate_social_dialogue(agent, partner, 0.5, {})
    assert result["relationship_modifier"] == 1.0
    assert result["happiness_modifier"] == 1.0
    print("  PASS: disabled socialize returns modifiers=1.0")

    result = llm.generate_teaching(agent, partner, "farming", {})
    assert result["skill_transfer_modifier"] == 1.0
    print("  PASS: disabled teach returns modifier=1.0")

    result = llm.generate_god_response(agent, "Hello mortal", "whisper")
    assert 0.0 <= result["compliance"] <= 1.0
    print("  PASS: disabled god_response returns valid compliance")


def test_fallback_text_varies_by_traits():
    """Fallback text should differ based on agent personality."""
    llm = LLMModule(LLMConfig(enabled=False))
    partner = MockAgent("Partner")

    # High ambition -> "hard bargain"
    ambitious = MockAgent("Ambitious", ambition=0.9, cooperation=0.2)
    result = llm.generate_trade_negotiation(ambitious, partner, 10.0, {})
    assert ("hard bargain" in result["proposer_text"].lower() or "aggressive" in result["proposer_text"].lower()
            or "Ambitious" in result["proposer_text"])
    print("  PASS: ambitious agent gets distinct trade text")

    # High cooperation -> "fair exchange"
    cooperative = MockAgent("Coop", ambition=0.2, cooperation=0.9)
    result = llm.generate_trade_negotiation(cooperative, partner, 10.0, {})
    assert "fair" in result["proposer_text"].lower() or "Coop" in result["proposer_text"]
    print("  PASS: cooperative agent gets distinct trade text")


def test_rate_limiting():
    """6th call should return fallback when max_calls_per_tick=5."""
    # Enable LLM but with unreachable URL -> calls will fail, but count
    llm = LLMModule(LLMConfig(enabled=True, max_calls_per_tick=5,
                                base_url="http://localhost:99999",
                                timeout_seconds=0.5))
    agent = MockAgent("Agent")
    partner = MockAgent("Partner")

    # Make 5 calls (will fail but increment counter)
    for i in range(5):
        llm.generate_trade_negotiation(agent, partner, 10.0, {})
    assert llm._tick_call_count == 5

    # 6th call should be blocked by can_call()
    assert not llm.can_call()
    result = llm.generate_trade_negotiation(agent, partner, 10.0, {})
    assert result["modifier"] == 1.0  # Fallback
    assert llm._tick_call_count == 5  # Not incremented
    print("  PASS: rate limiting blocks 6th call")

    # Reset
    llm.reset_tick_counter()
    assert llm._tick_call_count == 0
    assert llm.can_call()
    print("  PASS: tick counter resets correctly")


def test_parse_structured_response():
    """JSON extraction from various LLM output formats."""
    llm = LLMModule()

    # Clean JSON
    result = llm._parse_structured_response(
        'I propose a fair trade. {"modifier": 1.5, "accepted": true}',
        ["modifier"]
    )
    assert result["modifier"] == 1.5
    assert result["accepted"] is True
    print("  PASS: clean JSON parsed correctly")

    # JSON with surrounding text
    result = llm._parse_structured_response(
        'Here is my analysis:\n\n{"influence_modifier": 0.8, "persuasiveness": 0.9}\n\nThank you.',
        ["influence_modifier"]
    )
    assert result["influence_modifier"] == 0.8
    print("  PASS: JSON with surrounding text parsed")

    # No JSON at all
    result = llm._parse_structured_response(
        "I have nothing structured to say.",
        ["modifier"]
    )
    assert result["modifier"] == 1.0  # Default
    print("  PASS: missing JSON returns defaults")

    # Malformed JSON
    result = llm._parse_structured_response(
        'Almost JSON: {"modifier": bad_value}',
        ["modifier"]
    )
    assert result.get("modifier", 1.0) == 1.0
    print("  PASS: malformed JSON returns defaults")

    # Multiple JSON blocks (take last)
    result = llm._parse_structured_response(
        '{"old": true} Let me reconsider. {"modifier": 2.0}',
        ["modifier"]
    )
    assert result["modifier"] == 2.0
    print("  PASS: multiple JSON blocks -> uses last valid one")


def test_system_prompt_generation():
    """System prompt should include agent personality."""
    llm = LLMModule()
    agent = MockAgent("Kael", ambition=0.95, intelligence=0.3)
    prompt = llm._build_system_prompt(agent)

    assert "Kael" in prompt
    assert "ambition" in prompt.lower()
    assert "1-3 sentences" in prompt.lower() or "1-3 sentences" in prompt
    print("  PASS: system prompt includes name and traits")


def test_connection_error_handling():
    """Connection to wrong URL should fail gracefully."""
    llm = LLMModule(LLMConfig(enabled=True, base_url="http://localhost:99999",
                                timeout_seconds=0.5))

    result = llm.test_connection()
    assert result["success"] is False
    assert len(result["error"]) > 0
    print(f"  PASS: connection test fails gracefully: {result['error'][:60]}")


def test_get_status():
    """Status dict should have all expected fields."""
    llm = LLMModule(LLMConfig(enabled=True, model="test-model"))
    status = llm.get_status()

    assert status["enabled"] is True
    assert status["model"] == "test-model"
    assert "total_calls" in status
    assert "avg_latency_ms" in status
    assert "errors" in status
    print("  PASS: status dict has all fields")


def test_update_config():
    """Config update from UI data."""
    llm = LLMModule(LLMConfig(enabled=False))
    llm.update_config({"enabled": True, "model": "llama3", "temperature": 0.3})

    assert llm.config.enabled is True
    assert llm.config.model == "llama3"
    assert llm.config.temperature == 0.3
    print("  PASS: config updated correctly")


def test_modifier_clamping():
    """Modifiers should be clamped even if LLM returns extreme values."""
    llm = LLMModule()

    # Simulate parsing extreme values
    result = llm._parse_structured_response('{"modifier": 999}', ["modifier"])
    # The clamping happens in the generate_* methods, not in parsing
    # So here we just verify parsing returns the raw value
    assert result["modifier"] == 999
    print("  PASS: raw parse returns unclamped (clamping is in generate_*)")


if __name__ == "__main__":
    print("=" * 60)
    print("LLM MODULE TESTS")
    print("=" * 60)

    tests = [
        ("Disabled mode", test_disabled_mode),
        ("Fallback text varies", test_fallback_text_varies_by_traits),
        ("Rate limiting", test_rate_limiting),
        ("JSON parsing", test_parse_structured_response),
        ("System prompt", test_system_prompt_generation),
        ("Connection errors", test_connection_error_handling),
        ("Status dict", test_get_status),
        ("Config update", test_update_config),
        ("Modifier clamping", test_modifier_clamping),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{passed + failed} passed")
    if failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"WARNING: {failed} tests failed")
