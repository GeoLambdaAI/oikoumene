# Security Policy

Oikoumene is a research simulation, not a multi-tenant production service.
This document records the threat model the code is hardened against, the
threats that are explicitly **out of scope**, and how to report a security
issue responsibly.

## Threat model

The default deployment target is **a single researcher running the simulator
on their own workstation** and reaching it via `http://localhost:5000`. All
security defaults are tuned for that scenario.

### In scope (the code is hardened against these)

| Threat | Mitigation |
|---|---|
| Path traversal via crafted `run_id` in `/api/logger/download/<run_id>` | `run_id` is regex-validated to the `YYYYMMDD_HHMMSS` format produced by `sim_logger.py`; the resolved path is verified to stay under the repository's `logs/` directory ([app.py](app.py)). |
| Predictable Flask session-signing key | If `FLASK_SECRET_KEY` is unset, a per-process random key is generated (`secrets.token_hex(32)`) and a warning is printed ([app.py](app.py)). |
| Cross-origin requests driving the SocketIO API | `cors_allowed_origins` defaults to `http://localhost:5000` and `http://127.0.0.1:5000`. Override via `CORS_ALLOWED_ORIGINS` (comma-separated) only when fronted by an authenticated reverse proxy. |
| LAN exposure of the unauthenticated control surface | The server binds to `127.0.0.1` by default. `BIND_HOST=0.0.0.0` is only honoured when explicitly set, and a warning is printed in that case. |
| Leakage of LLM `api_key` to clients | `LLMModule.get_status()` returns provider / model / base_url but never the key. `world.get_full_state()` (sent over SocketIO and dumped to snapshots) inherits this whitelist. |

### Out of scope (the code does NOT defend against these)

| Threat | Why it's out of scope |
|---|---|
| Multi-user authentication / authorization | The simulator has no user concept. Every endpoint is unauthenticated by design. If you need multi-user access, place the app behind an authenticated reverse proxy (Caddy + `basic_auth`, oauth2-proxy, Cloudflare Access, etc.). |
| Rate limiting on HTTP routes | A local single user cannot meaningfully DoS themselves. Apply rate limiting at the reverse proxy layer if the simulator is exposed. |
| CSRF on SocketIO events | Mitigated implicitly by the localhost CORS default. If you broaden CORS, consider adding a CSRF token check on socket events. |
| Sandboxing of the LLM provider URL | The user controls `base_url`; the simulator will dutifully call whatever URL is configured. This is intentional for self-hosted Ollama or air-gapped Mistral deployments. Do not point this at untrusted hosts. |
| Hardening of `god_mode` interventions | God mode lets the operator drop drought, plague, technology, and direct messages into the simulation. It is a debugging / experimentation feature, not a privilege boundary. Anyone who can reach the SocketIO endpoint can use it. |

## Operator checklist before exposing the simulator beyond localhost

1. Set `FLASK_SECRET_KEY` to a stable random string (`python -c "import secrets; print(secrets.token_hex(32))"`).
2. Set `CORS_ALLOWED_ORIGINS` to your real frontend origin only.
3. Set `BIND_HOST` only after putting the app behind an authenticated reverse proxy.
4. Treat `logs/` and any `metadata.json` as potentially-disclosable artifacts (they contain seeds and scenario configs, but **not** API keys).
5. Verify your `requirements.txt` install resolves to the latest patch versions of `flask`, `flask-socketio`, and `eventlet` — these are the network-facing dependencies.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

**Preferred channel — GitHub private vulnerability reporting.**
Submit your report at
[github.com/GeoLambdaAI/world-genesis/security/advisories/new](https://github.com/GeoLambdaAI/world-genesis/security/advisories/new).
This gives you an authenticated, end-to-end private thread with the
maintainers, and the resulting advisory becomes citable once we publish
the fix.

**Cannot use GitHub?** Reach us through the contact form at
[www.geolambda.ai](https://www.geolambda.ai) and reference World Genesis
plus the word `security` in your message.

In either channel, please include:

- A clear description of the issue and the threat scenario.
- Reproduction steps (or a proof-of-concept).
- Affected version (Git commit hash if available).
- Whether you are willing to be credited in `CHANGELOG.md` once the issue is fixed.

You should expect an acknowledgement within **5 working days** and a triage
update within **15 working days**. We aim to ship a fix and coordinated
disclosure within 90 days for high-severity issues.

## Dependency hygiene

The simulator's network-facing surface comes from four packages:

- `flask` and `flask-socketio` — HTTP and WebSocket transport.
- `eventlet` — green-thread runtime under SocketIO. Has had several
  past CVEs; the lower bound `eventlet>=0.35.0` requires fixes from
  the 2024-09 release. Production deployments should pin to the latest
  patch.
- `requests` — outbound HTTP to LLM providers and data-source APIs.
  TLS verification uses the library default (`verify=True`).

`numpy`, `scipy`, `networkx`, `shapely` are computational dependencies
with no network surface.

## Acknowledgements

This security policy was drafted as part of the v0.1.0 hardening pass.
See [`CHANGELOG.md`](CHANGELOG.md) for the corresponding entries.
