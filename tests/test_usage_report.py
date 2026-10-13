from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from heart_risk import usage_report as report

DASHBOARD = str(Path(__file__).parent.parent / "dashboard.py")
NOW = pd.Timestamp("2026-10-13 15:00", tz=report.TIMEZONE)


def event(hours_ago, session, name, ref="direct", preset=None):
    return (NOW - pd.Timedelta(hours=hours_ago), session, name, ref, preset)


@pytest.fixture
def events():
    rows = [
        event(1, "a", "session_start", "resume"),
        event(1, "a", "preset_clicked", "resume", "High-risk 70-year-old"),
        event(1, "a", "answers_edited", "resume"),
        event(30, "b", "session_start", "github"),
        event(50, "c", "session_start", "resume"),
        event(50, "c", "preset_clicked", "resume", "Healthy 35-year-old"),
        event(3, "bot", "keepalive"),
        event(9, "bot2", "keepalive"),
    ]
    return report.prepare(pd.DataFrame(rows, columns=report.EVENT_COLUMNS))


def test_visits_exclude_bot_and_flag_engagement(events):
    visits = report.visits_table(events)
    assert list(visits["session_id"]) == ["a", "b", "c"]
    assert list(visits["clicked_preset"]) == [True, False, True]
    assert list(visits["edited_answers"]) == [True, False, False]
    assert list(visits["engaged"]) == [True, False, True]


def test_summary_and_sources(events):
    visits = report.visits_table(events)
    stats = report.summary(visits)
    assert stats == {"visits": 3, "engaged_rate": pytest.approx(2 / 3), "top_source": "resume"}

    sources = report.by_source(visits).set_index("ref")
    assert sources.loc["resume", "visits"] == 2
    assert sources.loc["resume", "engaged_rate"] == 1.0
    assert sources.loc["github", "engaged_rate"] == 0.0


def test_window_filter(events):
    visits = report.visits_table(events)
    last_day = report.in_window(visits, NOW - pd.Timedelta(days=1), NOW)
    assert list(last_day["session_id"]) == ["a"]


def test_daily_visits_fill_missing_days(events):
    visits = report.visits_table(events)
    daily = report.daily_visits(visits, NOW - pd.Timedelta(days=6), NOW)
    assert daily["day"].nunique() == 7
    assert daily["visits"].sum() == 3


def test_funnel_and_presets(events):
    visits = report.visits_table(events)
    assert list(report.funnel(visits)["visitors"]) == [3, 2, 1]
    presets = report.preset_clicks(events, visits["session_id"]).set_index("preset")["clicks"]
    assert presets.to_dict() == {"High-risk 70-year-old": 1, "Healthy 35-year-old": 1}


def test_keepalive_status(events):
    status = report.keepalive_status(events, NOW)
    assert status["hours_ago"] == pytest.approx(3)
    assert status["healthy"]
    assert not report.keepalive_status(events, NOW + pd.Timedelta(hours=20))["healthy"]


def test_empty_events_do_not_crash():
    empty = report.prepare(pd.DataFrame(columns=report.EVENT_COLUMNS))
    visits = report.visits_table(empty)
    assert report.summary(visits)["visits"] == 0
    assert report.keepalive_status(empty, NOW)["last"] is None


def test_dashboard_renders_sample_data_without_database():
    at = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not at.exception
    assert "sample data" in at.warning[0].value
    assert any(m.label == "Visits" for m in at.metric)

    at.selectbox[0].select("All time").run()
    assert not at.exception
