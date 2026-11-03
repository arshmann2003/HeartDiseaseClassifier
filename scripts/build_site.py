"""Assemble the static demo into _site/ for GitHub Pages.

The page source lives in web/. Evaluation figures stay in reports/figures,
where train.py writes them, and are copied in here. Analytics configuration
is injected at build time so the publishable Supabase key is not committed.
"""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"


def analytics_javascript() -> str:
    local = ROOT / "web" / "config.js"
    if local.exists():
        return local.read_text()

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if (not url or not key) and secrets.exists():
        block = tomllib.loads(secrets.read_text()).get("supabase", {})
        url = url or str(block.get("url", ""))
        key = key or str(block.get("key", ""))
    if url and key:
        payload = json.dumps({"url": url, "key": key})
        return f"window.ANALYTICS = {payload};\n"
    return "window.ANALYTICS = null;\n"


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(ROOT / "web", SITE)
    figures = SITE / "figures"
    figures.mkdir(exist_ok=True)
    for image in sorted((ROOT / "reports" / "figures").glob("*.png")):
        shutil.copy(image, figures / image.name)
    example = SITE / "config.example.js"
    if example.exists():
        example.unlink()
    (SITE / "config.js").write_text(analytics_javascript())
    print(f"Wrote {SITE}")


if __name__ == "__main__":
    main()
