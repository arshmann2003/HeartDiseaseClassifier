import time
from pathlib import Path

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from heart_risk.analytics import EventLogger, clean_ref

APP = str(Path(__file__).parent.parent / "app.py")


class FakeResponse:
    def __init__(self, status: int = 201):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def sent(monkeypatch):
    """Capture every request the app would send to Supabase."""
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers})
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    st.cache_resource.clear()
    yield calls
    st.cache_resource.clear()


def wait_for(calls, count, timeout=5.0):
    deadline = time.monotonic() + timeout
    while len(calls) < count and time.monotonic() < deadline:
        time.sleep(0.05)
    return [c["json"]["event"] for c in calls]


def run_app(query_params=None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.secrets["supabase"] = {"url": "https://example.supabase.co", "key": "sb_publishable_test"}
    for name, value in (query_params or {}).items():
        at.query_params[name] = value
    return at.run()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "direct"),
        ("", "direct"),
        ("Resume", "resume"),
        ("linked in!", "linkedin"),
        ("<script>", "script"),
    ],
)
def test_clean_ref(raw, expected):
    assert clean_ref(raw) == expected


def test_publishable_key_is_not_sent_as_bearer_token():
    logger = EventLogger("https://example.supabase.co/", "sb_publishable_abc")
    assert logger.endpoint == "https://example.supabase.co/rest/v1/events"
    assert "Authorization" not in logger.headers


def test_legacy_jwt_key_is_sent_as_bearer_token():
    logger = EventLogger("https://example.supabase.co", "eyJhbGciOi.legacy.key")
    assert logger.headers["Authorization"] == "Bearer eyJhbGciOi.legacy.key"


def test_failed_requests_are_dropped_not_raised():
    def failing_post(*args, **kwargs):
        raise requests.ConnectionError("network down")

    logger = EventLogger("https://example.supabase.co", "sb_publishable_abc", post=failing_post)
    assert logger.log("session", "session_start", "direct").result() is False


def test_visit_and_preset_click_are_logged_with_ref(sent):
    at = run_app({"ref": "resume"})
    next(b for b in at.button if b.label == "High-risk 70-year-old").click().run()

    events = wait_for(sent, 2)
    assert events == ["session_start", "preset_clicked"]
    assert {c["json"]["ref"] for c in sent} == {"resume"}
    assert sent[1]["json"]["properties"] == {"preset": "High-risk 70-year-old"}
    assert len({c["json"]["session_id"] for c in sent}) == 1


def test_health_answers_are_never_sent(sent):
    at = run_app()
    at.toggle(key="HighBP").set_value(True).run()
    at.toggle(key="Smoker").set_value(True).run()

    events = wait_for(sent, 2)
    assert events == ["session_start", "answers_edited"]
    assert all(c["json"]["properties"] == {} for c in sent)


def test_keepalive_visits_are_tagged_and_not_counted(sent):
    at = run_app({"source": "keepalive"})
    next(b for b in at.button if b.label == "Healthy 35-year-old").click().run()

    time.sleep(0.5)
    assert wait_for(sent, 1) == ["keepalive"]


def test_app_runs_without_analytics_configured():
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
