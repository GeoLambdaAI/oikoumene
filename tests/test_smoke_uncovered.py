"""Smoke tests for live code paths that neither the fingerprint runs nor any other test executed.

Safety net for the package refactor: each function is called with minimal valid
input and the result is checked for type and basic shape. The LLM is disabled,
so the LLM functions take their trait-based fallback path.
"""
import numpy as np
import pytest

from oikoumene.agents.agents import Agent, AgentMemory
from oikoumene.llm.llm_module import LLMConfig, LLMModule
from oikoumene.core.world import Settlement, World


@pytest.fixture()
def world() -> World:
    w = World(seed=1)
    w.logger.config.enabled = False
    return w


@pytest.fixture()
def llm() -> LLMModule:
    return LLMModule(LLMConfig(enabled=False))


def _agent(world: World) -> Agent:
    return Agent(10.0, 20.0, rng=world.rng)


def test_generate_trait_response(world: World, llm: LLMModule) -> None:
    out = llm._generate_trait_response(_agent(world), "Hello, how are you?")
    assert isinstance(out, dict) and {"text", "tone", "goal_change"} <= out.keys()
    assert isinstance(out["text"], str) and out["text"]


def test_generate_direct_chat(world: World, llm: LLMModule) -> None:
    out = llm.generate_direct_chat(_agent(world), "Hello")
    assert isinstance(out, dict) and {"text", "tone", "llm_used"} <= out.keys()
    assert isinstance(out["text"], str) and out["llm_used"] is False


def test_generate_group_chat(world: World, llm: LLMModule) -> None:
    agents = [_agent(world), _agent(world)]
    out = llm.generate_group_chat(agents, "Hello everyone")
    assert isinstance(out, list) and len(out) == 2
    assert [r["agent_id"] for r in out] == [a.id for a in agents]
    assert all({"agent_name", "text", "tone", "llm_used"} <= r.keys() for r in out)


def test_store_episode_appends_and_caps() -> None:
    memory = AgentMemory(capacity=2)
    for i in range(3):
        memory.store_episode({"event": i})
    assert isinstance(memory.episodic, list)
    assert [e["event"] for e in memory.episodic] == [1, 2]


def test_action_research(world: World) -> None:
    out = _agent(world)._action_research(world)
    assert isinstance(out, dict) and {"success", "reward", "description"} <= out.keys()
    assert isinstance(out["success"], bool) and isinstance(out["description"], str)


def test_action_govern_with_audience(world: World) -> None:
    world.agents = [Agent(10.0 + 0.1 * i, 20.0, rng=world.rng) for i in range(4)]
    world._rebuild_spatial_grid()
    out = world.agents[0]._action_govern(world)
    assert isinstance(out, dict) and out["success"] is True
    assert isinstance(out["reward"], float) and "Governed 3" in out["description"]


def test_settlement_to_dict(world: World) -> None:
    out = Settlement(10.0, 20.0, founder_id=1, rng=np.random.RandomState(0)).to_dict()
    assert isinstance(out, dict)
    assert {"id", "name", "lat", "lng", "population", "governance", "culture", "treasury", "age"} <= out.keys()
    assert out["lat"] == 10.0 and out["lng"] == 20.0
