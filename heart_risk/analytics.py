"""Anonymous usage events, stored in a Supabase table through its REST API.

Only interaction events are recorded (visits, example profiles clicked, whether
answers were edited). Health answers and risk estimates are never sent.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor

import requests

log = logging.getLogger(__name__)

MAX_REF_LENGTH = 40


def clean_ref(value: str | None) -> str:
    """Normalize the `?ref=` link tag so arbitrary input can't pollute the table."""
    if not value:
        return "direct"
    ref = re.sub(r"[^a-z0-9_-]", "", str(value).lower())[:MAX_REF_LENGTH]
    return ref or "direct"


class EventLogger:
    """Sends events in a background thread so the app never waits on the network.

    Failures are logged and dropped: analytics must never break the app.
    """

    def __init__(self, url: str, key: str, table: str = "events", timeout: float = 3.0, post=None):
        self.endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
        self.headers = {"apikey": key, "Content-Type": "application/json", "Prefer": "return=minimal"}
        if not key.startswith("sb_"):
            # Legacy anon keys are JWTs and must also be sent as a bearer token.
            self.headers["Authorization"] = f"Bearer {key}"
        self.timeout = timeout
        self._post = post or requests.post
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analytics")

    def log(self, session_id: str, event: str, ref: str, properties: dict | None = None) -> Future:
        payload = {"session_id": session_id, "event": event, "ref": ref, "properties": properties or {}}
        return self._executor.submit(self._send, payload)

    def _send(self, payload: dict) -> bool:
        try:
            response = self._post(self.endpoint, json=payload, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return True
        except Exception as exc:
            log.warning("Analytics event dropped: %s", exc)
            return False
