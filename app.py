"""
Flask + SocketIO server for World Genesis.
Provides real-time WebSocket updates and a zoomable world map interface.
"""

import eventlet
eventlet.monkey_patch()

import os
import re
import secrets
import time

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

from world import World

# Run-ID format produced by sim_logger.py: 8-digit date + underscore +
# 6-digit time (e.g. 20260414_161426). Anchors path-traversal defence
# in download_historical_log below.
_RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Flask session signing key. If FLASK_SECRET_KEY is not provided we
# generate a fresh random key for this process — sessions then do not
# survive a restart, but that is strictly safer than a shared predictable
# default that would let anyone forge cookies against deployments where
# the operator forgot to set the variable.
_secret_key = os.environ.get("FLASK_SECRET_KEY")
if not _secret_key:
    _secret_key = secrets.token_hex(32)
    print("[security] FLASK_SECRET_KEY unset — using ephemeral random key for this process.")
app.config["SECRET_KEY"] = _secret_key

# CORS: restrict to localhost by default. Override via CORS_ALLOWED_ORIGINS
# (comma-separated) when fronted by an authenticated reverse proxy.
_cors_env = os.environ.get("CORS_ALLOWED_ORIGINS", "").strip()
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else ["http://localhost:5000", "http://127.0.0.1:5000"]
)
socketio = SocketIO(app, cors_allowed_origins=_cors_origins, async_mode="eventlet")

# Global simulation state
world: World = None
sim_running = False
sim_speed = 1.0  # Ticks per second multiplier
sim_thread = None


def create_world(seed: int = 42, initial_agents: int = 25,
                  start_year_bp: int = 70000, scenario_id: str = "historical"):
    global world
    from scenarios import SCENARIOS
    scenario = SCENARIOS.get(scenario_id, SCENARIOS["historical"])

    print(f"  Scenario: {scenario.name}")
    print("  Generating Earth terrain...")
    world = World(seed=seed, cell_size_deg=2.0,
                  config={"start_year_bp": scenario.start_year_bp},
                  scenario_id=scenario_id)
    print(f"  Spawning {initial_agents} agents...")
    world.spawn_initial_agents(initial_agents)
    print(f"  World ready: {len(world.agents)} agents, "
          f"{len(world.geopolitics.nations)} nations")
    return world


def simulation_loop():
    """Background simulation loop."""
    global sim_running
    while sim_running:
        if world:
            stats = world.step()
            socketio.emit("tick", stats)

            # Send full state every 10 ticks for sync
            if world.tick % 10 == 0:
                socketio.emit("full_state", world.get_full_state())

        delay = max(0.02, 1.0 / max(0.1, sim_speed))
        time.sleep(delay)


# ============================================================================
# Routes
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    if world is None:
        create_world()
    return jsonify(world.get_full_state())


@app.route("/api/macro")
def get_macro():
    if world is None:
        return jsonify({})
    return jsonify(world.macro.get_summary())


@app.route("/api/geopolitics")
def get_geopolitics():
    if world is None:
        return jsonify({})
    return jsonify({
        "nations": world.geopolitics.get_nations_list(),
        "conflicts": world.geopolitics.active_conflicts,
        "summary": world.geopolitics.get_summary(),
    })


@app.route("/api/logger/status")
def get_logger_status():
    if world is None:
        return jsonify({"enabled": False})
    return jsonify(world.logger.get_status())


@app.route("/api/logger/runs")
def get_logger_runs():
    if world is None:
        return jsonify([])
    return jsonify(world.logger.list_runs())


@app.route("/api/logger/download")
def download_log_csv():
    """Download the current run's CSV file."""
    if world is None:
        return "No simulation running", 404
    csv_path = world.logger.get_csv_path()
    if csv_path is None:
        return "No log file available", 404
    from flask import send_file
    return send_file(csv_path, mimetype="text/csv",
                     as_attachment=True,
                     download_name=f"simulation_{world.logger._run_id}.csv")


@app.route("/api/logger/download/<run_id>")
def download_historical_log(run_id):
    """Download a previous run's CSV.

    Security: ``run_id`` must match the timestamp format that
    ``sim_logger.py`` produces. The resolved path is then verified to
    stay under the repository's ``logs/`` directory, which blocks
    traversal via crafted URLs *and* via symlink escape inside that
    directory.
    """
    if not _RUN_ID_PATTERN.match(run_id):
        return "Invalid run id", 400

    from pathlib import Path
    logs_root = (Path(__file__).resolve().parent / "logs").resolve()
    csv_path = (logs_root / run_id / "timeseries.csv").resolve()

    if logs_root not in csv_path.parents:
        return "Invalid run id", 400
    if not csv_path.exists():
        return "Run not found", 404

    from flask import send_file
    return send_file(str(csv_path), mimetype="text/csv",
                     as_attachment=True,
                     download_name=f"simulation_{run_id}.csv")


@app.route("/api/scenarios")
def get_scenarios():
    from scenarios import SCENARIOS
    return jsonify({
        sid: {"name": s.name, "description": s.description, "start_date": s.start_date}
        for sid, s in SCENARIOS.items()
    })


@app.route("/api/llm/status")
def get_llm_status():
    if world is None:
        return jsonify({"enabled": False})
    return jsonify(world.llm.get_status())


@app.route("/api/god/status")
def get_god_status():
    if world is None:
        return jsonify({"enabled": False})
    return jsonify(world.god_mode.get_status())


@app.route("/api/god/log")
def get_god_log():
    if world is None:
        return jsonify([])
    return jsonify(world.god_mode.get_intervention_log())


@app.route("/api/jepa/status")
def get_jepa_status():
    if world is None:
        from world import _torch_available
        return jsonify({"backend": "numpy", "preset": "default",
                        "torch_available": _torch_available()})
    return jsonify(world.get_jepa_status())


@app.route("/api/dialogues")
def get_dialogues():
    if world is None:
        return jsonify([])
    return jsonify(world.recent_dialogues[-50:])


@app.route("/api/agent/<int:agent_id>")
def get_agent(agent_id):
    if world is None:
        return jsonify({"error": "No world"}), 404
    for a in world.agents:
        if a.id == agent_id:
            return jsonify(a.to_dict())
    return jsonify({"error": "Agent not found"}), 404


# ============================================================================
# SocketIO Events
# ============================================================================

@socketio.on("connect")
def on_connect():
    global world
    if world is None:
        create_world()
    socketio.emit("full_state", world.get_full_state())


@socketio.on("start")
def on_start():
    global sim_running, sim_thread
    if not sim_running:
        sim_running = True
        sim_thread = eventlet.spawn(simulation_loop)

    socketio.emit("status", {"running": True})


@socketio.on("stop")
def on_stop():
    global sim_running
    sim_running = False
    socketio.emit("status", {"running": False})


@socketio.on("step")
def on_step():
    if world:
        stats = world.step()
        socketio.emit("tick", stats)
        socketio.emit("full_state", world.get_full_state())


@socketio.on("set_speed")
def on_set_speed(data):
    global sim_speed
    sim_speed = float(data.get("speed", 1.0))


@socketio.on("reset")
def on_reset(data=None):
    global sim_running, world
    sim_running = False
    time.sleep(0.1)
    seed = data.get("seed", 42) if data else 42
    scenario = data.get("scenario", "historical") if data else "historical"
    agents = data.get("agents", 25) if data else 25
    if scenario == "present_day" and agents == 25:
        agents = 300
    create_world(seed=seed, initial_agents=agents, scenario_id=scenario)
    socketio.emit("full_state", world.get_full_state())
    socketio.emit("status", {"running": False})


# ---- LLM Control ----
@socketio.on("set_llm_config")
def on_set_llm_config(data):
    if world:
        world.llm.update_config(data)
        socketio.emit("llm_status", world.llm.get_status())


@socketio.on("test_llm")
def on_test_llm():
    if world:
        result = world.llm.test_connection()
        socketio.emit("llm_test_result", result)


# ---- JEPA World-Model Backend ----
@socketio.on("set_jepa_backend")
def on_set_jepa_backend(data):
    """Swap the shared JEPA backend/preset at runtime.

    Rebuilding the model and repointing every agent must not interleave with
    a tick, so the sim loop is paused for the swap (cooperative eventlet
    scheduling means this is the same safety pattern used by on_reset) and
    resumed afterwards only if it was running and the swap succeeded.
    """
    global sim_running, sim_thread
    if not world:
        return
    backend = data.get("backend", "numpy")
    preset = data.get("preset", "default")
    device = data.get("device", "auto")

    was_running = sim_running
    if was_running:
        sim_running = False
        time.sleep(0.12)  # let the current tick finish before swapping

    result = world.set_jepa_backend(backend=backend, preset=preset, device=device)

    if was_running:
        sim_running = True
        sim_thread = eventlet.spawn(simulation_loop)

    socketio.emit("jepa_status", result)


# ---- God Mode ----
@socketio.on("set_god_mode")
def on_set_god_mode(data):
    if world:
        world.god_mode.config.enabled = data.get("enabled", False)
        socketio.emit("god_mode_status", world.god_mode.get_status())


@socketio.on("god_whisper")
def on_god_whisper(data):
    if not world:
        return
    agent_id = data.get("agent_id", 0)
    message = data.get("message", "")

    result = world.god_mode.whisper_to_agent(world, agent_id, message)

    # Find the agent and process the whisper immediately (don't wait for next tick)
    agent = None
    for a in world.agents:
        if a.id == agent_id and a.alive:
            agent = a
            break

    if agent and message and agent.divine_messages:
        # Process the whisper now so we can return the reaction
        msg = agent.divine_messages.pop(0)
        world.god_mode._process_divine_message(agent, msg, world)

        result["agent_response"] = agent.last_dialogue or ""
        result["agent_tone"] = "neutral"
        result["complied"] = agent.memory.episodic[-1].get("complied", False) if agent.memory.episodic else False
        result["goal_changed"] = agent.memory.episodic[-1].get("goal_parsed") if agent.memory.episodic else None
        result["divine_trust"] = round(agent.divine_trust, 2)
        result["used_llm"] = hasattr(agent, 'last_dialogue') and bool(agent.last_dialogue)

    socketio.emit("god_result", result)


@socketio.on("god_vision")
def on_god_vision(data):
    if world:
        result = world.god_mode.send_vision_to_nation(
            world, data.get("nation_id", 0), data.get("message", ""))
        socketio.emit("god_result", result)


@socketio.on("god_commandment")
def on_god_commandment(data):
    if not world:
        return
    message = data.get("message", "")
    result = world.god_mode.issue_commandment(
        world, message,
        data.get("lat", 0), data.get("lng", 0), data.get("radius", 10))

    # Process all queued messages immediately so agents react now
    complied_count = 0
    refused_count = 0
    goals_changed = {}
    for agent in world.agents:
        if not agent.alive:
            continue
        if hasattr(agent, 'divine_messages') and agent.divine_messages:
            msg = agent.divine_messages.pop(0)
            world.god_mode._process_divine_message(agent, msg, world)
            # Check what happened
            if agent.memory.episodic:
                last = agent.memory.episodic[-1]
                if last.get("type") == "divine_message":
                    if last.get("complied"):
                        complied_count += 1
                        goal = last.get("goal_parsed")
                        if goal:
                            goals_changed[goal] = goals_changed.get(goal, 0) + 1
                    else:
                        refused_count += 1

    result["event_type"] = "commandment"
    result["message"] = message[:100]
    result["lat"] = data.get("lat", 0)
    result["lng"] = data.get("lng", 0)
    result["radius"] = data.get("radius", 10)
    result["complied"] = complied_count
    result["refused"] = refused_count
    result["goals_changed"] = goals_changed
    socketio.emit("god_event_result", result)


@socketio.on("god_event")
def on_god_event(data):
    if not world:
        return
    event_type = data.get("type", "")
    if event_type == "drought":
        result = world.god_mode.trigger_drought(
            world, data.get("lat", 0), data.get("lng", 0),
            data.get("radius", 5), data.get("severity", 0.5),
            data.get("duration", 50))
    elif event_type == "plague":
        result = world.god_mode.trigger_plague(
            world, data.get("lat", 0), data.get("lng", 0),
            data.get("radius", 5), data.get("severity", 0.3),
            data.get("duration", 30))
    elif event_type == "discovery":
        result = world.god_mode.trigger_resource_discovery(
            world, data.get("lat", 0), data.get("lng", 0),
            data.get("resource", "food"), data.get("amount", 50))
    elif event_type == "tech":
        result = world.god_mode.grant_technology(
            world, agent_id=data.get("agent_id"),
            skill=data.get("skill", "research"), boost=data.get("boost", 0.2))
    elif event_type == "climate":
        result = world.god_mode.modify_climate(
            world, temperature_delta=data.get("temp_delta", 0),
            co2_delta=data.get("co2_delta", 0))
    else:
        result = {"success": False, "error": f"Unknown event type: {event_type}"}
    result["event_type"] = event_type
    socketio.emit("god_event_result", result)


# ---- Chat ----
@socketio.on("chat_with_agent")
def on_chat_with_agent(data):
    """Direct chat with a single agent by ID."""
    if not world or not world.llm:
        socketio.emit("chat_response", {"error": "LLM not available"})
        return
    agent_id = data.get("agent_id")
    message = data.get("message", "")
    context = data.get("context", "as_peer")

    agent = None
    for a in world.agents:
        if a.id == agent_id and a.alive:
            agent = a
            break
    if not agent:
        socketio.emit("chat_response", {"error": "Agent not found"})
        return

    ws = world.get_local_state(agent.lat, agent.lng)
    resp = world.llm.generate_direct_chat(agent, message, context, ws)

    agent.last_dialogue = resp.get("text")
    agent.dialogue_history.append({
        "tick": world.tick, "partner": "USER",
        "text": resp.get("text", ""), "user_said": message,
    })
    if len(agent.dialogue_history) > 10:
        agent.dialogue_history.pop(0)

    socketio.emit("chat_response", {
        "agent_id": agent.id,
        "agent_name": agent.name,
        "text": resp.get("text", ""),
        "tone": resp.get("tone", "neutral"),
        "goal_change": resp.get("goal_change"),
        "llm_used": resp.get("llm_used", False),
        "error": resp.get("error", ""),
    })


@socketio.on("chat_with_group")
def on_chat_with_group(data):
    """Chat with multiple agents (by IDs or nearby selected agent)."""
    if not world or not world.llm:
        socketio.emit("group_chat_response", {"error": "LLM not available"})
        return
    agent_ids = data.get("agent_ids", [])
    message = data.get("message", "")
    context = data.get("context", "as_peer")

    agents = [a for a in world.agents if a.id in agent_ids and a.alive]
    if not agents:
        socketio.emit("group_chat_response", {"error": "No agents found"})
        return

    responses = world.llm.generate_group_chat(agents, message, context)

    for resp, agent in zip(responses, agents):
        agent.last_dialogue = resp.get("text")
        agent.dialogue_history.append({
            "tick": world.tick, "partner": "USER",
            "text": resp.get("text", ""), "user_said": message,
        })
        if len(agent.dialogue_history) > 10:
            agent.dialogue_history.pop(0)

    socketio.emit("group_chat_response", {"responses": responses})


@socketio.on("select_agent")
def on_select_agent(data):
    agent_id = data.get("id")
    if world and agent_id:
        for a in world.agents:
            if a.id == agent_id:
                socketio.emit("agent_detail", a.to_dict())
                return
        socketio.emit("agent_detail", None)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Default to loopback so the unauthenticated control surface is not
    # exposed on the LAN. Set BIND_HOST=0.0.0.0 explicitly only when the
    # server is fronted by an authenticated reverse proxy.
    host = os.environ.get("BIND_HOST", "127.0.0.1")
    create_world()
    display_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print("=" * 60)
    print("  World Genesis — Earth")
    print(f"  Open http://{display_host}:{port} in your browser")
    if host == "0.0.0.0":
        print("  WARNING: bound to 0.0.0.0 — reachable from the network.")
        print("           Endpoints are not authenticated; place behind a")
        print("           reverse proxy with auth before exposing publicly.")
    print("=" * 60)
    socketio.run(app, host=host, port=port, debug=False)
