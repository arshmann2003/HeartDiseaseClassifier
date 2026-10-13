"""Private analytics dashboard for the hosted demo. Run it on your own machine:

    streamlit run dashboard.py --server.port 8502

It reads from Supabase with the read-only `analytics_reader` login configured
under [analytics_db] in .streamlit/secrets.toml. Without that, it shows sample
data so the layout can be previewed.
"""

import altair as alt
import pandas as pd
import streamlit as st

from heart_risk import usage_report as report

APP_URL = "https://heart-risk-screener.streamlit.app"
WINDOWS = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": None}
QUERY = """
    select created_at, session_id::text as session_id, event, ref, properties ->> 'preset' as preset
    from public.events
    order by created_at
"""

st.set_page_config(page_title="Demo analytics", layout="wide")


def db_config() -> dict | None:
    try:
        return dict(st.secrets["analytics_db"])
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner="Loading events...")
def load_events(config: dict) -> pd.DataFrame:
    import psycopg

    with psycopg.connect(**config, sslmode="require", connect_timeout=10) as conn, conn.cursor() as cur:
        cur.execute(QUERY)
        return pd.DataFrame(cur.fetchall(), columns=[column.name for column in cur.description])


def get_events(now: pd.Timestamp) -> tuple[pd.DataFrame, bool]:
    config = db_config()
    if config is None:
        return report.sample_events(now), True
    try:
        return load_events(config), False
    except Exception as exc:
        st.error(f"Could not connect to Supabase: {exc}")
        st.stop()


def daily_chart(daily: pd.DataFrame):
    return (
        alt.Chart(daily)
        .mark_bar()
        .encode(
            x=alt.X("yearmonthdate(day):T", title=None, axis=alt.Axis(format="%b %d")),
            y=alt.Y("sum(visits):Q", title="Visits"),
            color=alt.Color("ref:N", title="Source", legend=alt.Legend(orient="top")),
            tooltip=[
                alt.Tooltip("yearmonthdate(day):T", title="Day", format="%a %b %d"),
                alt.Tooltip("ref:N", title="Source"),
                alt.Tooltip("sum(visits):Q", title="Visits"),
            ],
        )
        .properties(height=280)
    )


def horizontal_bars(data: pd.DataFrame, label: str, value: str, title: str, color: str = "#e8554e"):
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadius=3, color=color)
        .encode(
            x=alt.X(f"{value}:Q", title=title),
            y=alt.Y(f"{label}:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
            tooltip=[alt.Tooltip(f"{label}:N", title=""), alt.Tooltip(f"{value}:Q", title=title)],
        )
    )
    text = bars.mark_text(align="left", dx=4, color="#333").encode(text=f"{value}:Q")
    return (bars + text).properties(height=max(140, 48 * len(data)))


def main() -> None:
    now = pd.Timestamp.now(tz=report.TIMEZONE)
    events, is_sample = get_events(now)
    events = report.prepare(events)
    all_visits = report.visits_table(events)

    head, controls = st.columns([3, 1], vertical_alignment="bottom")
    head.title("Demo analytics")
    head.caption(f"Usage of [{APP_URL.removeprefix('https://')}]({APP_URL}). Keep-alive bot visits excluded.")
    window_name = controls.selectbox("Time window", list(WINDOWS), index=1)
    if controls.button("Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    if is_sample:
        st.warning(
            "Not connected to Supabase, so this is **sample data**. Add the `[analytics_db]` login "
            "to `.streamlit/secrets.toml` to see real visits (setup in `supabase/reader_role.sql`)."
        )

    days = WINDOWS[window_name]
    if days is None:
        first = all_visits["created_at"].min() if len(all_visits) else now
        start, previous = first, None
    else:
        start = now - pd.Timedelta(days=days)
        previous = report.in_window(all_visits, start - pd.Timedelta(days=days), start)
    visits = report.in_window(all_visits, None if days is None else start, now)
    stats = report.summary(visits)
    keepalive = report.keepalive_status(events, now)

    daily = report.daily_visits(visits, start, now)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Visits",
        stats["visits"],
        delta=None if previous is None else stats["visits"] - len(previous),
        delta_description=None if previous is None else "vs prior",
        chart_data=daily.groupby("day")["visits"].sum().tolist(),
        chart_type="area",
        border=True,
    )
    k2.metric(
        "Interacted",
        f"{stats['engaged_rate']:.0%}",
        help="Share of visitors who clicked an example profile or entered their own answers",
        border=True,
    )
    k3.metric("Top source", stats["top_source"] or "–", border=True)
    if keepalive["last"] is None:
        k4.metric("Keep-alive", "No runs yet", border=True)
    else:
        k4.metric(
            "Keep-alive",
            "Healthy" if keepalive["healthy"] else "Late",
            delta=f"last run {keepalive['hours_ago']:.1f} h ago",
            delta_color="normal" if keepalive["healthy"] else "inverse",
            delta_arrow="off",
            border=True,
        )

    if visits.empty:
        st.info("No visits in this period yet.")
        return

    with st.container(border=True):
        st.markdown("**Visits per day by source**")
        st.altair_chart(daily_chart(daily), width="stretch")

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Where visitors come from**")
        sources = report.by_source(visits)
        st.altair_chart(horizontal_bars(sources, "ref", "visits", "Visits"), width="stretch")
        st.dataframe(
            sources.rename(
                columns={"ref": "Source", "visits": "Visits", "engaged": "Interacted", "engaged_rate": "Rate"}
            ),
            column_config={
                "Rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)
            },
            hide_index=True,
            width="stretch",
        )
    with right, st.container(border=True):
        st.markdown("**What visitors do**")
        st.altair_chart(
            horizontal_bars(report.funnel(visits), "step", "visitors", "Visitors", color="#4c9be8"),
            width="stretch",
        )
        presets = report.preset_clicks(events, visits["session_id"])
        if not presets.empty:
            st.markdown("**Example profiles clicked**")
            st.altair_chart(horizontal_bars(presets, "preset", "clicks", "Clicks"), width="stretch")

    with st.container(border=True):
        st.markdown("**Recent visits**")
        recent = visits.head(25).assign(visited=lambda d: d["created_at"].dt.strftime("%a %b %d, %-I:%M %p"))
        st.dataframe(
            recent[["visited", "ref", "clicked_preset", "edited_answers"]].rename(
                columns={
                    "visited": "Visited (Vancouver time)",
                    "ref": "Source",
                    "clicked_preset": "Clicked a profile",
                    "edited_answers": "Entered answers",
                }
            ),
            hide_index=True,
            width="stretch",
        )


main()
