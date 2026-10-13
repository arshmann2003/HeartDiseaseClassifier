"""Turn raw usage events into the numbers shown on the analytics dashboard.

All functions take the events table as a DataFrame with columns
`created_at, session_id, event, ref, preset`, so they can be tested without a
database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMEZONE = "America/Vancouver"
KEEPALIVE_EXPECTED_HOURS = 6
EVENT_COLUMNS = ["created_at", "session_id", "event", "ref", "preset"]


def prepare(events: pd.DataFrame) -> pd.DataFrame:
    df = events.reindex(columns=EVENT_COLUMNS).copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(TIMEZONE)
    return df


def visits_table(events: pd.DataFrame) -> pd.DataFrame:
    """One row per real visit, with what that visitor did."""
    visits = events.loc[events["event"] == "session_start", ["created_at", "session_id", "ref"]]
    sessions_with = {
        name: set(events.loc[events["event"] == name, "session_id"])
        for name in ("preset_clicked", "answers_edited")
    }
    return (
        visits.assign(
            clicked_preset=visits["session_id"].isin(sessions_with["preset_clicked"]),
            edited_answers=visits["session_id"].isin(sessions_with["answers_edited"]),
        )
        .assign(engaged=lambda d: d["clicked_preset"] | d["edited_answers"])
        .sort_values("created_at", ascending=False)
        .reset_index(drop=True)
    )


def in_window(visits: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> pd.DataFrame:
    mask = visits["created_at"] < end
    if start is not None:
        mask &= visits["created_at"] >= start
    return visits[mask]


def summary(visits: pd.DataFrame) -> dict:
    count = len(visits)
    return {
        "visits": count,
        "engaged_rate": float(visits["engaged"].mean()) if count else 0.0,
        "top_source": visits["ref"].value_counts().idxmax() if count else None,
    }


def daily_visits(visits: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Visits per day and source, with zero-filled days so the chart has no gaps."""
    days = pd.date_range(start.normalize(), end.normalize(), freq="D", tz=TIMEZONE).date
    counts = (
        visits.assign(day=visits["created_at"].dt.date)
        .groupby(["day", "ref"])
        .size()
        .rename("visits")
        .reset_index()
    )
    refs = counts["ref"].unique() if len(counts) else np.array(["direct"])
    grid = pd.MultiIndex.from_product([days, refs], names=["day", "ref"]).to_frame(index=False)
    daily = grid.merge(counts, on=["day", "ref"], how="left").fillna({"visits": 0})
    daily["visits"] = daily["visits"].astype(int)
    daily["day"] = pd.to_datetime(daily["day"])
    return daily


def by_source(visits: pd.DataFrame) -> pd.DataFrame:
    return (
        visits.groupby("ref")
        .agg(visits=("session_id", "size"), engaged=("engaged", "sum"))
        .assign(engaged_rate=lambda d: d["engaged"] / d["visits"])
        .sort_values("visits", ascending=False)
        .reset_index()
    )


def funnel(visits: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": ["Visited", "Clicked a profile", "Entered own answers"],
            "visitors": [
                len(visits),
                int(visits["clicked_preset"].sum()),
                int(visits["edited_answers"].sum()),
            ],
        }
    )


def preset_clicks(events: pd.DataFrame, session_ids) -> pd.DataFrame:
    clicks = events[(events["event"] == "preset_clicked") & events["session_id"].isin(session_ids)]
    return clicks["preset"].value_counts().rename_axis("preset").reset_index(name="clicks")


def keepalive_status(events: pd.DataFrame, now: pd.Timestamp) -> dict:
    """Whether the scheduled keep-alive visit is reaching the app on time."""
    times = events.loc[events["event"] == "keepalive", "created_at"]
    if times.empty:
        return {"last": None, "hours_ago": None, "healthy": False}
    last = times.max()
    hours_ago = (now - last).total_seconds() / 3600
    # GitHub can delay scheduled runs, so allow one missed run before warning.
    return {"last": last, "hours_ago": hours_ago, "healthy": hours_ago <= 2 * KEEPALIVE_EXPECTED_HOURS}


def sample_events(now: pd.Timestamp, days: int = 45, seed: int = 0) -> pd.DataFrame:
    """Realistic fake events for previewing the dashboard before it is connected."""
    rng = np.random.default_rng(seed)
    refs = ["resume", "github", "linkedin", "direct", "amazon"]
    ref_weights = [0.35, 0.25, 0.2, 0.15, 0.05]
    presets = ["Healthy 35-year-old", "55-year-old with high BP", "High-risk 70-year-old"]
    rows = []
    for day in range(days):
        day_start = now.normalize() - pd.Timedelta(days=days - day - 1)
        for _ in range(rng.poisson(1.5 + 3 * day / days)):
            at = day_start + pd.Timedelta(minutes=int(rng.integers(8 * 60, 23 * 60)))
            if at > now:
                continue
            session, ref = f"sample-{len(rows)}", str(rng.choice(refs, p=ref_weights))
            rows.append((at, session, "session_start", ref, None))
            if rng.random() < 0.6:
                preset = str(rng.choice(presets, p=[0.25, 0.3, 0.45]))
                rows.append((at, session, "preset_clicked", ref, preset))
            if rng.random() < 0.35:
                rows.append((at, session, "answers_edited", ref, None))
    for hours in range(0, days * 24, KEEPALIVE_EXPECTED_HOURS):
        at = now - pd.Timedelta(hours=hours + 1)
        rows.append((at, f"bot-{hours}", "keepalive", "direct", None))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
