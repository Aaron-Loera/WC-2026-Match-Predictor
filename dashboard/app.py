"""
Streamlit public dashboard for the WC2026 Match Predictor.

Displays tournament odds, per-match win/draw/loss predictions, and a historical
odds tracker. All data is fetched from the FastAPI layer (`api/main.py`) over
HTTP — this module has no direct ML imports, mirroring the separation of
concerns described for Module 6 of the implementation plan.

Pages:

1. Tournament Odds: Bar chart of win probability per team, top-8 favourites
   highlighted. Sourced from `GET /predictions/tournament`.

2. Match Predictor: Team dropdowns for any group-stage pairing, with
   horizontal probability bars for win/draw/loss. Sourced from
   `GET /predictions/matches`.

3. Odds Tracker: Line chart of tournament win probability over time per team.
   Sourced from `GET /predictions/history`.

API responses are cached via `st.cache_data` (TTL configurable through the
`DASHBOARD_CACHE_TTL` env var, default 3600s — drop to 300s on active
matchdays for near-live updates). The sidebar surfaces `GET /health` so users
can see at a glance whether the API is reachable and how fresh the underlying
predictions are.
"""
from __future__ import annotations
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CACHE_TTL_SECONDS = int(os.getenv("DASHBOARD_CACHE_TTL", "3600"))
REQUEST_TIMEOUT = 60  # Render's free tier can take ~50s to wake from sleep

st.set_page_config(page_title="WC2026 Match Predictor", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _get(endpoint: str) -> dict | list:
    """
    GET an endpoint from the prediction API and return the parsed JSON body.

    Args:
        endpoint: Path beginning with "/", e.g. "/predictions/tournament".

    Returns:
        dict | list: Parsed JSON response body.
    """
    response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_tournament_odds() -> dict:
    """Cached wrapper around `GET /predictions/tournament`."""
    return _get("/predictions/tournament")


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_match_predictions() -> list[dict]:
    """Cached wrapper around `GET /predictions/matches`."""
    return _get("/predictions/matches")


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_odds_history() -> list[dict]:
    """Cached wrapper around `GET /predictions/history`."""
    return _get("/predictions/history")


@st.cache_data(ttl=60)
def fetch_health() -> dict:
    """Cached wrapper around `GET /health` — short TTL so status stays current."""
    return _get("/health")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def render_tournament_odds() -> None:
    """Home page: bar chart of tournament win probability, top 8 highlighted."""
    st.title("🏆 Tournament Odds")
    st.caption("Win probability per team from the latest Monte Carlo simulation.")

    try:
        payload = fetch_tournament_odds()
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
        return

    odds = payload["tournament_odds"]
    df = (
        pd.DataFrame(odds.items(), columns=["team", "win_probability"])
        .sort_values("win_probability", ascending=False)
        .reset_index(drop=True)
    )
    df["rank"] = df.index + 1
    df["tier"] = df["rank"].apply(lambda r: "Top 8" if r <= 8 else "Other")

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("win_probability:Q", title="Tournament win probability", axis=alt.Axis(format="%")),
            y=alt.Y("team:N", sort="-x", title=None),
            color=alt.Color(
                "tier:N",
                scale=alt.Scale(domain=["Top 8", "Other"], range=["#f4a300", "#4c78a8"]),
                legend=alt.Legend(title=None),
            ),
            tooltip=["team", alt.Tooltip("win_probability:Q", title="Win probability", format=".1%")],
        )
        .properties(height=28 * len(df))
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Generated {payload['generated_at']} · {payload['n_simulations']:,} simulations")


def render_match_predictor() -> None:
    """Match Predictor page: team dropdowns + win/draw/loss probability bars."""
    st.title("🔮 Match Predictor")
    st.caption("Win / draw / loss probabilities for any group-stage pairing.")

    try:
        matches = fetch_match_predictions()
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
        return

    df = pd.DataFrame(matches)

    col1, col2 = st.columns(2)
    home = col1.selectbox("Home team", sorted(df["home"].unique()))
    away_options = sorted(df.loc[df["home"] == home, "away"].unique())
    away = col2.selectbox("Away team", away_options)

    match = df[(df["home"] == home) & (df["away"] == away)].iloc[0]

    st.subheader(f"{home} vs {away}")
    st.write(f"**{home} win**")
    st.progress(match["p_home_win"], text=f"{match['p_home_win']:.1%}")
    st.write("**Draw**")
    st.progress(match["p_draw"], text=f"{match['p_draw']:.1%}")
    st.write(f"**{away} win**")
    st.progress(match["p_away_win"], text=f"{match['p_away_win']:.1%}")


def render_odds_tracker() -> None:
    """Odds Tracker page: line chart of each team's tournament win % over time."""
    st.title("📈 Odds Tracker")
    st.caption("Tournament win probability over time, one snapshot per prediction re-export.")

    try:
        history = fetch_odds_history()
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
        return

    if len(history) < 2:
        st.info("Not enough snapshots yet to chart a trend — check back after the next retrain.")
        return

    long_rows = [
        {"generated_at": snapshot["generated_at"], "team": team, "win_probability": prob}
        for snapshot in history
        for team, prob in snapshot["tournament_odds"].items()
    ]
    df = pd.DataFrame(long_rows)
    df["generated_at"] = pd.to_datetime(df["generated_at"])

    latest_odds = history[-1]["tournament_odds"]
    default_teams = sorted(latest_odds, key=latest_odds.get, reverse=True)[:8]
    teams = st.multiselect("Teams to track", sorted(df["team"].unique()), default=default_teams)

    if not teams:
        st.info("Select at least one team to plot.")
        return

    chart = (
        alt.Chart(df[df["team"].isin(teams)])
        .mark_line(point=True)
        .encode(
            x=alt.X("generated_at:T", title="Snapshot"),
            y=alt.Y("win_probability:Q", title="Tournament win probability", axis=alt.Axis(format="%")),
            color=alt.Color("team:N", title="Team"),
            tooltip=["team", "generated_at:T", alt.Tooltip("win_probability:Q", title="Win probability", format=".1%")],
        )
        .properties(height=500)
    )
    st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar — navigation + API health
# ---------------------------------------------------------------------------

PAGES = {
    "🏆 Tournament Odds": render_tournament_odds,
    "🔮 Match Predictor": render_match_predictor,
    "📈 Odds Tracker": render_odds_tracker,
}


def render_sidebar() -> str:
    """
    Render sidebar navigation and an API health indicator.

    Returns:
        str: The selected page label, used to look up the render function in `PAGES`.
    """
    st.sidebar.title("WC2026 Predictor")
    page = st.sidebar.radio("Navigate", list(PAGES.keys()))

    st.sidebar.divider()
    try:
        health = fetch_health()
        if health.get("status") == "ok" and health.get("model_loaded") and health.get("predictions_loaded"):
            st.sidebar.success("API online")
        else:
            st.sidebar.warning("API degraded")
        if health.get("generated_at"):
            st.sidebar.caption(f"Predictions generated {health['generated_at']}")
    except requests.RequestException:
        st.sidebar.error("API unreachable")

    return page


def main() -> None:
    page = render_sidebar()
    PAGES[page]()


if __name__ == "__main__":
    main()
