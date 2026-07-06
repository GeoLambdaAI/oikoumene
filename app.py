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

# Concurrency control (eventlet cooperative green-threads).
#
# world_lock serializes every mutation of the shared World so a tick that yields
# mid-step (LLM HTTP calls yield under eventlet) cannot interleave with an event
# handler that also mutates the world (step/reset/god-event/chat).
#
# loop_generation invalidates stale simulation-loop greenlets: each spawned loop
# captures the current generation and exits as soon as it no longer matches.
# Without it, an old loop still parked in time.sleep() (up to ~10 s at low speed)
# would resume and run alongside a freshly spawned loop after a reset/restart
# flipped sim_running back to True — two loops stepping the same World.
world_lock = eventlet.semaphore.Semaphore(1)
loop_generation = 0


def create_world(seed: int = 42, initial_agents: int = 25,
                  start_year_bp: int = 70000, scenario_id: str = "historical"):
    global world
    from scenarios import SCENARIOS
    scenario = SCENARIOS.get(scenario_id, SCENARIOS["historical"])

    # Finalize the outgoing world's logger before discarding it, so its CSV is
    # flushed/closed and metadata gets ended_at/total_rows (otherwise every
    # reset leaked an open file handle and a truncated, never-closed run).
    if world is not None:
        try:
            world.logger.end_run()
        except Exception as exc:  # never let logging teardown block a reset
            print(f"[logger] end_run on reset failed: {exc}")

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


def simulation_loop(my_generation):
    """Background simulation loop.

    Runs only while it owns the current generation; a reset/restart bumps
    loop_generation, so any older loop exits at its next check instead of
    double-stepping the world.
    """
    while sim_running and my_generation == loop_generation:
        if world:
            with world_lock:
                # Re-check inside the lock: a reset may have run while we waited.
                if not (sim_running and my_generation == loop_generation) or world is None:
                    break
                stats = world.step()
                emit_full = world.tick % 10 == 0
                full_state = world.get_full_state() if emit_full else None
            socketio.emit("tick", stats)
            if emit_full:
                socketio.emit("full_state", full_state)

        delay = max(0.02, 1.0 / max(0.1, sim_speed))
        time.sleep(delay)


def _start_loop():
    """Start a fresh simulation-loop greenlet, invalidating any prior one."""
    global sim_running, sim_thread, loop_generation
    loop_generation += 1
    sim_running = True
    sim_thread = eventlet.spawn(simulation_loop, loop_generation)


def _stop_loop():
    """Signal the running loop (if any) to exit and invalidate its generation."""
    global sim_running, loop_generation
    sim_running = False
    loop_generation += 1


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
    if not sim_running:
        _start_loop()
    socketio.emit("status", {"running": True})


@socketio.on("stop")
def on_stop():
    _stop_loop()
    socketio.emit("status", {"running": False})


@socketio.on("step")
def on_step():
    if world:
        with world_lock:
            if world is None:
                return
            stats = world.step()
            full_state = world.get_full_state()
        socketio.emit("tick", stats)
        socketio.emit("full_state", full_state)


@socketio.on("set_speed")
def on_set_speed(data):
    global sim_speed
    try:
        speed = float((data or {}).get("speed", 1.0))
    except (TypeError, ValueError):
        return
    # Reject NaN/inf and clamp to a sane range; the loop also clamps its delay.
    if speed != speed or speed in (float("inf"), float("-inf")):
        return
    sim_speed = min(100.0, max(0.1, speed))


# Bounds for client-supplied reset parameters.
MAX_INITIAL_AGENTS = 5000
_SEED_MAX = 2 ** 32 - 1


@socketio.on("reset")
def on_reset(data=None):
    # Stop and invalidate any running loop so it cannot resume against the new
    # world after we rebuild it.
    _stop_loop()
    data = data if isinstance(data, dict) else {}

    from scenarios import SCENARIOS
    scenario = data.get("scenario", "historical")
    if not isinstance(scenario, str) or scenario not in SCENARIOS:
        scenario = "historical"

    try:
        seed = int(data.get("seed", 42))
    except (TypeError, ValueError):
        seed = 42
    seed = seed % (_SEED_MAX + 1)  # numpy RandomState requires 0..2**32-1

    raw_agents = data.get("agents", 25)
    try:
        agents = int(raw_agents)
    except (TypeError, ValueError):
        agents = 25
    if scenario == "present_day" and agents == 25:
        agents = 300
    agents = min(MAX_INITIAL_AGENTS, max(1, agents))

    # Rebuild under the lock so a not-yet-exited loop greenlet can never observe
    # a half-constructed world.
    with world_lock:
        create_world(seed=seed, initial_agents=agents, scenario_id=scenario)
        full_state = world.get_full_state()
    socketio.emit("full_state", full_state)
    socketio.emit("status", {"running": False})


# ---- LLM Control ----
@socketio.on("set_llm_config")
def on_set_llm_config(data):
    if world:
        result = world.llm.update_config(data if isinstance(data, dict) else {})
        status = world.llm.get_status()
        status["config_errors"] = result.get("errors", [])
        socketio.emit("llm_status", status)


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
    if not world:
        return
    data = data if isinstance(data, dict) else {}
    backend = data.get("backend", "numpy")
    preset = data.get("preset", "default")
    device = data.get("device", "auto")

    was_running = sim_running
    if was_running:
        _stop_loop()  # invalidate the current loop's generation

    # The lock guarantees the swap does not interleave with an in-flight tick.
    with world_lock:
        result = world.set_jepa_backend(backend=backend, preset=preset, device=device)

    if was_running:
        _start_loop()

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

    # Hold the lock across processing (which may call the LLM and yield) so the
    # world cannot be reset/swapped underneath this handler mid-flight.
    with world_lock:
        if world is None:
            return
        result = world.god_mode.whisper_to_agent(world, agent_id, message)

        # Find the agent and process the whisper immediately (don't wait for tick)
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
        with world_lock:
            if world is None:
                return
            result = world.god_mode.send_vision_to_nation(
                world, data.get("nation_id", 0), data.get("message", ""))
        socketio.emit("god_result", result)


@socketio.on("god_commandment")
def on_god_commandment(data):
    if not world:
        return
    message = data.get("message", "")
    # Hold the lock across per-agent LLM processing (which yields) so the world
    # cannot be reset/swapped mid-flight.
    with world_lock:
        if world is None:
            return
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
    # Serialize the state mutation against the tick loop so a god event applies
    # cleanly between ticks rather than mid-step.
    with world_lock:
        if world is None:
            return
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

    with world_lock:
        if world is None:
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

    with world_lock:
        if world is None:
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

def main() -> None:
    """Console entry point: launch the World Genesis web server.

    Exposed as the ``world-genesis`` script via [project.scripts] so the package
    is runnable after ``pip install`` as well as via ``python app.py``.
    """
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


if __name__ == "__main__":
    main()
