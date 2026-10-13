"""Visit the live Streamlit demo so it does not go to sleep.

Streamlit Community Cloud only counts browser sessions as activity, so this
opens the app in headless Chromium, clicks the wake-up button if the app is
already asleep, and waits until the app has rendered.

    python scripts/keep_alive.py [URL]

Exits non-zero if the app does not load, so a failing scheduled run doubles
as an outage alert.
"""

import re
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

# The source tag lets the app log this as a keep-alive visit instead of a real visitor.
DEFAULT_URL = "https://heart-risk-screener.streamlit.app/?source=keepalive"
APP_TEXT = "Heart Disease Risk Screener"
LOAD_TIMEOUT_S = 180


def app_rendered(page) -> bool:
    for frame in page.frames:
        try:
            if frame.get_by_text(APP_TEXT, exact=True).count() > 0:
                return True
        except Exception:
            continue
    return False


def main(url: str) -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)

        wake = page.get_by_role("button", name=re.compile("get this app back up", re.I))
        try:
            wake.click(timeout=15_000)
            print("App was asleep; clicked the wake-up button.")
        except PlaywrightTimeout:
            print("App was already awake.")

        deadline = time.monotonic() + LOAD_TIMEOUT_S
        while time.monotonic() < deadline:
            if app_rendered(page):
                print("App loaded.")
                page.wait_for_timeout(10_000)
                browser.close()
                return 0
            page.wait_for_timeout(5_000)

        page.screenshot(path="keep_alive_failure.png", full_page=True)
        browser.close()
        print(f"App did not load within {LOAD_TIMEOUT_S}s.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL))
