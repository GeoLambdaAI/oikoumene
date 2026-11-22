"""
base_url validation for the LLM endpoint (moved from llm_module.py): SSRF and
API-key-exfiltration hardening via validate_base_url.
"""

import ipaddress
import os
import socket
from typing import Optional
from urllib.parse import urlparse


# ============================================================================
# base_url validation (SSRF / API-key-exfiltration hardening)
# ============================================================================
#
# The LLM endpoint (`base_url`) is operator-configurable at runtime, including
# via the unauthenticated SocketIO control surface. Two abuses must be blocked:
#
#   1. API-key exfiltration: pointing base_url at an attacker host so the
#      configured Bearer token is sent there. Mitigated by (a) refusing to keep
#      an api_key across a base_url change unless it is re-supplied together
#      (see LLMModule.update_config), and (b) this egress guard.
#   2. SSRF: using the server as a proxy to reach cloud-metadata / internal
#      services (e.g. http://169.254.169.254/...).
#
# Default policy blocks link-local (incl. the 169.254.169.254 metadata IP) and
# other non-routable/reserved ranges. Loopback and private ranges stay allowed
# because a local Ollama or a LAN model server is the primary, documented use
# case — and with the api_key no longer following a base_url change, a redirect
# to those ranges can no longer carry the operator's credential.
#
# For stricter deployments, set OIKOUMENE_LLM_ALLOWED_HOSTS to a
# comma-separated hostname allowlist; when set, ONLY those hosts are permitted.

_METADATA_HOSTS = {
    "169.254.169.254",       # AWS/GCP/Azure/OpenStack IMDS
    "metadata.google.internal",
    "metadata",
}


def _env_host_allowlist() -> Optional[set]:
    raw = os.environ.get("OIKOUMENE_LLM_ALLOWED_HOSTS", "").strip()
    if not raw:
        return None
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def validate_base_url(url: str) -> tuple:
    """
    Validate an LLM base_url for SSRF safety.

    Returns (ok: bool, reason: str). reason is "" when ok.
    """
    if not isinstance(url, str) or not url.strip():
        return False, "base_url must be a non-empty string"

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return False, f"unsupported scheme: {parsed.scheme or '(none)'}"
    host = parsed.hostname
    if not host:
        return False, "base_url has no host"
    host_l = host.lower()

    allow = _env_host_allowlist()
    if allow is not None:
        return (True, "") if host_l in allow else (False, f"host not in allowlist: {host_l}")

    if host_l in _METADATA_HOSTS:
        return False, "cloud-metadata host is blocked"

    # Resolve the host and reject any address in a link-local / reserved /
    # multicast / unspecified range. Loopback and private ranges are allowed.
    try:
        port = parsed.port  # may raise ValueError for an out-of-range port
    except ValueError:
        port = None
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        # Unresolvable host: let the request itself fail later rather than
        # guessing. Not an SSRF risk (no connection can be made).
        return True, ""
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError:
            continue
        # Loopback (127.0.0.0/8, ::1) is always allowed — it's the local-Ollama
        # case. Note ::1 has is_reserved == True in Python, so it MUST be
        # exempted before the block check below (else `localhost` is rejected on
        # any host that resolves it to IPv6 first, e.g. CI runners).
        if ip.is_loopback:
            continue
        if ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
            return False, f"blocked address range: {addr}"
    return True, ""
