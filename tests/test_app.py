from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")


def estimated_risk(at: AppTest) -> float:
    metric = next(m for m in at.metric if m.label == "Estimated risk")
    return float(metric.value.rstrip("%"))


def test_app_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Heart Disease Risk Screener"


def test_presets_change_the_estimate():
    at = AppTest.from_file(APP, default_timeout=30).run()

    next(b for b in at.button if b.label == "Healthy 35-year-old").click().run()
    healthy = estimated_risk(at)

    next(b for b in at.button if b.label == "High-risk 70-year-old").click().run()
    high_risk = estimated_risk(at)

    assert not at.exception
    assert high_risk > healthy
