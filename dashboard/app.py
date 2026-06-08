"""
Streamlit public dashboard for the WC2026 Match Predictor.

Displays tournament odds, per-match win/draw/loss predictions, and a historical
odds tracker. All data is fetched from the FastAPI layer (`api/main.py`) over
HTTP — this module has no direct ML imports, mirroring the separation of
concerns described for Module 6 of the implementation plan.

Pages:

1. Tournament Odds: KPI strip (favourite, simulations, field size, kickoff
   countdown) above a ranked, flag-annotated bar list of each team's title
   probability, top-8 contenders highlighted in gold. Sourced from
   `GET /predictions/tournament`.

2. Match Predictor: Team selectors (with flags) for any group-stage pairing, a
   single split win/draw/loss probability bar, three outcome cards, and a
   most-likely-result callout. Sourced from `GET /predictions/matches`.

3. Odds Tracker: Biggest riser/faller cards plus a multi-line chart of each
   team's title probability over time. Sourced from `GET /predictions/history`.

Theme: a dark "data-journalism" look — charcoal background, lighter grey cards,
and a pitch-green + gold accent palette with country flags for every team. The
colour scheme is enforced two ways: `.streamlit/config.toml` themes the native
Streamlit widgets, while the CSS injected here styles the custom cards and bars.
Team flags are small images served from flagcdn.com, keyed off the FIFA-name ->
ISO-code lookup below. The brand mark and browser tab both use the bundled
green soccer-ball icon (`assets/ball_icon.png`).

API responses are cached via `st.cache_data` (TTL configurable through the
`DASHBOARD_CACHE_TTL` env var, default 3600s — drop to 300s on active
matchdays for near-live updates). The header surfaces `GET /health` so users
can see at a glance whether the API is reachable and how fresh the underlying
predictions are.
"""
from __future__ import annotations

import datetime as dt
import html
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CACHE_TTL_SECONDS = int(os.getenv("DASHBOARD_CACHE_TTL", "3600"))
REQUEST_TIMEOUT = 60  # Render's free tier can take ~50s to wake from sleep

# Opening match of the tournament, used for the kickoff countdown KPI.
KICKOFF_DATE = dt.date(2026, 6, 11)

# --- Dark theme palette (single source of truth for the custom HTML) --------
BG = "#1B1D1F"            # charcoal app background
CARD = "#26292C"         # lighter grey card / surface
CARD_BD = "#34383C"      # card border
TRACK = "#303438"        # empty bar track
TXT = "#E8EAEC"          # primary text
TXT_BRIGHT = "#F4F5F6"   # headings / values
TXT2 = "#9AA0A6"         # secondary text
TXT3 = "#7A8087"         # tertiary text / hints
GREEN = "#7FB83E"        # chasing-pack bars, home-win fills
GREEN_TXT = "#9FD45B"    # bright green for figures on dark cards
GREEN_TINT = "#2A3A18"   # callout / active-nav fill
GREEN_TINT_TXT = "#CDE6A8"
GREEN_PILL_TXT = "#A6D86B"
GREEN_LINE = "#3A5A22"   # centre-line motif
GOLD = "#F2A93B"         # top-8 contender bars, away-win fills
GOLD_TXT = "#F2B85C"     # away-win figures
GREY = "#80868C"         # draw fills
RED = "#E5705B"          # faller accent
STATUS_OK = "#3BD68B"
STATUS_WARN = "#F2A93B"
STATUS_ERR = "#E5705B"

# Bundled brand/favicon icon; fall back to an emoji if the asset is missing.
_ICON_PATH = Path(__file__).resolve().parent / "assets" / "ball_icon.svg"
PAGE_ICON = str(_ICON_PATH) if _ICON_PATH.exists() else "⚽"

st.set_page_config(
    page_title="WC 2026 Match Predictor",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def ball_svg(size: int = 22) -> str:
    """Return the green soccer-ball brand mark as an inline SVG at `size` px.

    Mirrors `assets/ball_icon.svg` (ring + inscribed pentagon + spokes on a
    vertical green gradient) so the inline brand mark and the bundled icon
    stay visually identical.
    """
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 582 582" '
        'xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;flex:none;">'
        '<defs><linearGradient id="ballGradient" gradientUnits="userSpaceOnUse" '
        'x1="291" y1="32" x2="291" y2="550">'
        '<stop offset="0" stop-color="#9EDB35"/>'
        '<stop offset="1" stop-color="#5DA10C"/>'
        '</linearGradient></defs>'
        '<circle cx="291" cy="291" r="259" fill="none" stroke="url(#ballGradient)" stroke-width="63"/>'
        '<path d="M 291,148.5 L 426.5,247.0 L 374.8,406.3 L 207.2,406.3 L 155.5,247.0 Z" '
        'fill="none" stroke="url(#ballGradient)" stroke-width="63" stroke-linejoin="round"/>'
        '<g stroke="url(#ballGradient)" stroke-width="63">'
        '<line x1="291" y1="148.5" x2="291" y2="32"/>'
        '<line x1="426.5" y1="247.0" x2="537.3" y2="211.0"/>'
        '<line x1="374.8" y1="406.3" x2="443.2" y2="500.5"/>'
        '<line x1="207.2" y1="406.3" x2="138.8" y2="500.5"/>'
        '<line x1="155.5" y1="247.0" x2="44.7" y2="211.0"/>'
        '</g></svg>'
    )


# ---------------------------------------------------------------------------
# Team reference data: ISO codes (for flags) and confederations
# ---------------------------------------------------------------------------

# FIFA display name -> ISO 3166-1 alpha-2 (lowercase) for flagcdn. England and
# Scotland use flagcdn's UK-subdivision codes so they get their own flags.
TEAM_ISO: dict[str, str] = {
    "Mexico": "mx", "South Africa": "za", "Korea Republic": "kr", "Czechia": "cz",
    "Canada": "ca", "Bosnia and Herzegovina": "ba", "Qatar": "qa", "Switzerland": "ch",
    "Brazil": "br", "Morocco": "ma", "Haiti": "ht", "Scotland": "gb-sct",
    "USA": "us", "Paraguay": "py", "Australia": "au", "Turkey": "tr",
    "Germany": "de", "Curaçao": "cw", "Ivory Coast": "ci", "Ecuador": "ec",
    "Netherlands": "nl", "Japan": "jp", "Sweden": "se", "Tunisia": "tn",
    "Belgium": "be", "Egypt": "eg", "IR Iran": "ir", "New Zealand": "nz",
    "Spain": "es", "Cape Verde": "cv", "Saudi Arabia": "sa", "Uruguay": "uy",
    "France": "fr", "Senegal": "sn", "Iraq": "iq", "Norway": "no",
    "Argentina": "ar", "Algeria": "dz", "Austria": "at", "Jordan": "jo",
    "Portugal": "pt", "Congo DR": "cd", "Uzbekistan": "uz", "Colombia": "co",
    "England": "gb-eng", "Croatia": "hr", "Ghana": "gh", "Panama": "pa",
}

TEAM_CONFED: dict[str, str] = {
    # UEFA
    "Czechia": "UEFA", "Switzerland": "UEFA", "Germany": "UEFA", "Netherlands": "UEFA",
    "Sweden": "UEFA", "Belgium": "UEFA", "Spain": "UEFA", "France": "UEFA",
    "Norway": "UEFA", "Austria": "UEFA", "Portugal": "UEFA", "England": "UEFA",
    "Croatia": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA", "Bosnia and Herzegovina": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Paraguay": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Argentina": "CONMEBOL", "Colombia": "CONMEBOL",
    # CONCACAF
    "Mexico": "CONCACAF", "Canada": "CONCACAF", "Haiti": "CONCACAF",
    "USA": "CONCACAF", "Curaçao": "CONCACAF", "Panama": "CONCACAF",
    # CAF
    "South Africa": "CAF", "Morocco": "CAF", "Ivory Coast": "CAF", "Tunisia": "CAF",
    "Egypt": "CAF", "Cape Verde": "CAF", "Senegal": "CAF", "Algeria": "CAF",
    "Congo DR": "CAF", "Ghana": "CAF",
    # AFC
    "Korea Republic": "AFC", "Qatar": "AFC", "Australia": "AFC", "Japan": "AFC",
    "IR Iran": "AFC", "Saudi Arabia": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC",
    # OFC
    "New Zealand": "OFC",
}


def flag_img(team: str, height: int = 22) -> str:
    """
    Return an <img> tag for a team's flag, or a neutral placeholder if unknown.

    Args:
        team: FIFA display name, e.g. "Argentina".
        height: Rendered flag height in px (width is auto-scaled 4:3).

    Returns:
        str: HTML <img> (or <span> fallback) sized and styled for inline use.
    """
    iso = TEAM_ISO.get(team)
    width = round(height * 4 / 3)
    style = (
        f"width:{width}px;height:{height}px;border-radius:3px;object-fit:cover;"
        "border:0.5px solid rgba(255,255,255,0.25);vertical-align:middle;flex:none;"
    )
    if not iso:
        return f'<span style="{style}background:#3A3E42;display:inline-block;"></span>'
    return f'<img src="https://flagcdn.com/h40/{iso}.png" alt="" style="{style}">'


def confed(team: str) -> str:
    """Return a team's confederation acronym, or an empty string if unknown."""
    return TEAM_CONFED.get(team, "")


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------

def inject_theme() -> None:
    """Inject the dashboard's dark CSS once per run (palette, spacing, nav pills)."""
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2rem; padding-left: 3rem;
            padding-right: 3rem; max-width: 1500px; }
          #MainMenu, footer { visibility: hidden; }
          header[data-testid="stHeader"] { display: none !important; }
          [data-testid="stDecoration"] { display: none !important; }

          .wc-header { display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 12px; }
          .wc-brand { display: flex; align-items: center; gap: 9px;
            font-size: 16px; font-weight: 500; color: #F4F5F6; }
          .wc-status { display: flex; align-items: center; gap: 6px;
            font-size: 13px; color: #9AA0A6; }
          .wc-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

          .wc-rule { position: relative; height: 0;
            border-bottom: 1.5px solid #3A5A22; margin: 6px 0 22px; }
          .wc-rule::after { content: ""; position: absolute; left: 50%; top: -6px;
            width: 12px; height: 12px; transform: translateX(-50%);
            border: 1.5px solid #3A5A22; border-radius: 50%; background: #1B1D1F; }

          .wc-h1 { font-size: 26px; font-weight: 500; margin: 0 0 3px; color: #F4F5F6; }
          .wc-eyebrow { font-size: 14px; color: #9AA0A6; margin: 0 0 22px; }

          /* Nav pills (styled st.button row) */
          div[data-testid="stHorizontalBlock"] .stButton button {
            border: none; background: transparent; color: #9AA0A6;
            font-weight: 500; font-size: 14px; border-radius: 8px; padding: 7px 16px; }
          div[data-testid="stHorizontalBlock"] .stButton button:hover {
            background: #26292C; color: #E8EAEC; }
          div[data-testid="stHorizontalBlock"] .stButton button[kind="primary"] {
            background: #2A3A18; color: #A6D86B; }

          /* KPI + content cards */
          .wc-kpis { display: grid; grid-template-columns: repeat(4, 1fr);
            gap: 14px; margin-bottom: 26px; }
          .wc-kpi { background: #26292C; border-radius: 10px; padding: 16px 18px; }
          .wc-kpi .lbl { font-size: 13px; color: #9AA0A6; margin-bottom: 10px; }
          .wc-kpi .val { font-size: 30px; font-weight: 500; color: #F4F5F6;
            display: flex; align-items: center; gap: 9px; }
          .wc-kpi .sub { font-size: 13px; color: #7A8087; margin-top: 4px; }

          .wc-section { display: flex; align-items: center;
            justify-content: space-between; margin: 4px 0 14px; }
          .wc-section h3 { font-size: 18px; font-weight: 500; margin: 0; color: #F4F5F6; }
          .wc-legend { display: flex; gap: 16px; font-size: 13px; color: #9AA0A6; }
          .wc-legend span { display: flex; align-items: center; gap: 6px; }
          .wc-swatch { width: 11px; height: 11px; border-radius: 2px; }

          .wc-row { display: flex; align-items: center; gap: 12px; padding: 6px 0; }
          .wc-rank { width: 20px; text-align: right; font-size: 13px;
            color: #7A8087; font-variant-numeric: tabular-nums; }
          .wc-team { width: 170px; }
          .wc-team .nm { font-size: 15px; line-height: 1.25; color: #E8EAEC; }
          .wc-team .cf { font-size: 12px; color: #7A8087; }
          .wc-track { flex: 1; height: 24px; background: #303438;
            border-radius: 6px; overflow: hidden; }
          .wc-fill { height: 100%; border-radius: 6px; }
          .wc-pct { width: 56px; text-align: right; font-size: 15px;
            font-weight: 500; color: #F4F5F6; font-variant-numeric: tabular-nums; }
          .wc-divider { display: flex; align-items: center; gap: 8px;
            margin: 10px 0 8px; font-size: 12px; color: #7A8087; }
          .wc-divider .ln { flex: 1; border-top: 1px dashed #34383C; }

          .wc-foot { margin-top: 16px; font-size: 13px; color: #7A8087;
            border-top: 0.5px solid #303438; padding-top: 12px; }

          /* Match predictor */
          .wc-split { display: flex; height: 44px; border-radius: 9px;
            overflow: hidden; margin-bottom: 8px; }
          .wc-split div { display: flex; align-items: center; justify-content: center;
            font-size: 16px; font-weight: 500; }
          .wc-splitlbl { display: flex; justify-content: space-between;
            font-size: 12px; color: #7A8087; margin-bottom: 24px; }
          .wc-cards { display: grid; grid-template-columns: repeat(3, 1fr);
            gap: 14px; margin-bottom: 20px; }
          .wc-card { background: #26292C; border: 0.5px solid #34383C;
            border-radius: 12px; padding: 18px; }
          .wc-card .hd { display: flex; align-items: center; gap: 9px;
            margin-bottom: 12px; font-size: 14px; color: #9AA0A6; }
          .wc-card .big { font-size: 34px; font-weight: 500; }
          .wc-card .mini { height: 6px; background: #303438; border-radius: 3px;
            margin-top: 10px; overflow: hidden; }
          .wc-callout { display: flex; align-items: center; gap: 11px;
            background: #2A3A18; border-radius: 9px; padding: 14px 16px;
            font-size: 14px; color: #CDE6A8; }
          .wc-context { display: flex; align-items: center; gap: 8px;
            justify-content: center; margin: 0 0 20px; font-size: 13px; color: #9AA0A6; }

          /* Odds tracker */
          .wc-mover { background: #26292C; border-radius: 10px; padding: 16px 18px; }
          .wc-mover .lbl { font-size: 13px; color: #9AA0A6; margin-bottom: 10px;
            display: flex; align-items: center; gap: 6px; }
          .wc-mover .bd { display: flex; align-items: center; gap: 10px;
            font-size: 17px; font-weight: 500; color: #F4F5F6; }
          .wc-chips { display: flex; flex-wrap: wrap; gap: 9px; margin: 4px 0; }
          .wc-chip { display: flex; align-items: center; gap: 8px;
            background: #303438; border-radius: 16px; padding: 6px 11px 6px 8px;
            font-size: 14px; color: #E8EAEC; }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
# Shared chrome: header + top navigation
# ---------------------------------------------------------------------------

PAGE_LABELS = ["Tournament odds", "Match predictor", "Odds tracker"]


def _format_generated(raw: str | None) -> str:
    """Render an ISO timestamp as a friendly 'Jun 7, 2026 20:33 UTC' string."""
    if not raw:
        return "unknown"
    try:
        ts = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return ts.strftime("%b %-d, %Y %H:%M UTC")
    except (ValueError, TypeError):
        return raw


def render_header() -> None:
    """Render the brand row and an API health chip (replaces the old sidebar)."""
    dot, label = STATUS_WARN, "API status unknown"
    try:
        health = fetch_health()
        if (
            health.get("status") == "ok"
            and health.get("model_loaded")
            and health.get("predictions_loaded")
        ):
            dot, label = STATUS_OK, "API live"
        else:
            dot, label = STATUS_WARN, "API degraded"
        gen = _format_generated(health.get("generated_at"))
        label = f"{label} · predictions {gen}"
    except requests.RequestException:
        dot, label = STATUS_ERR, "API unreachable"

    st.markdown(
        f"""
        <div class="wc-header">
          <div class="wc-brand">{ball_svg(22)} WC 2026 Predictor</div>
          <div class="wc-status"><span class="wc-dot" style="background:{dot};"></span>{html.escape(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_nav() -> str:
    """
    Render the top pill navigation and return the active page label.

    Selection persists in `st.session_state["page"]`; the active pill is drawn
    with Streamlit's "primary" button styling (themed green via CSS).
    """
    if "page" not in st.session_state:
        st.session_state["page"] = PAGE_LABELS[0]

    cols = st.columns([1, 1, 1, 4])
    for label, col in zip(PAGE_LABELS, cols[:3]):
        is_active = st.session_state["page"] == label
        if col.button(label, key=f"nav_{label}", type="primary" if is_active else "secondary"):
            st.session_state["page"] = label

    st.markdown('<div class="wc-rule"></div>', unsafe_allow_html=True)
    return st.session_state["page"]


def page_intro(title: str, subtitle: str, eyebrow: str = "") -> None:
    """Render a page title + subtitle, with an optional right-aligned eyebrow date."""
    right = f'<span style="font-size:13px;color:{TXT3};">{html.escape(eyebrow)}</span>' if eyebrow else ""
    st.markdown(
        f'<div class="wc-section" style="margin-top:0;"><h3 class="wc-h1">{html.escape(title)}</h3>{right}</div>'
        f'<p class="wc-eyebrow">{html.escape(subtitle)}</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 1 — Tournament Odds
# ---------------------------------------------------------------------------

def render_tournament_odds() -> None:
    """KPI strip + ranked, flag-annotated bar list with top-8 contenders in gold."""
    try:
        payload = fetch_tournament_odds()
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
        return

    odds = payload["tournament_odds"]
    df = (
        pd.DataFrame(odds.items(), columns=["team", "p"])
        .sort_values("p", ascending=False)
        .reset_index(drop=True)
    )
    n_sims = payload.get("n_simulations", 0)
    gen_full = _format_generated(payload.get("generated_at"))
    gen_date = " ".join(gen_full.split(" ")[0:3])

    page_intro("Who wins the World Cup?",
               "Each team's probability of lifting the trophy, across "
               f"{n_sims:,} simulated tournaments.",
               eyebrow=gen_date)

    fav = df.iloc[0]
    days_to_kickoff = (KICKOFF_DATE - dt.date.today()).days
    kickoff_val = "Underway" if days_to_kickoff < 0 else (
        "Today" if days_to_kickoff == 0 else f"{days_to_kickoff} days")
    st.markdown(
        f"""
        <div class="wc-kpis">
          <div class="wc-kpi"><div class="lbl">Favourite</div>
            <div class="val">{flag_img(fav['team'])} {fav['p'] * 100:.1f}%</div>
            <div class="sub">{html.escape(fav['team'])}</div></div>
          <div class="wc-kpi"><div class="lbl">Simulations</div>
            <div class="val">{n_sims:,}</div><div class="sub">Monte Carlo runs</div></div>
          <div class="wc-kpi"><div class="lbl">Field</div>
            <div class="val">{len(df)}</div><div class="sub">teams qualified</div></div>
          <div class="wc-kpi"><div class="lbl">Kick-off</div>
            <div class="val">{kickoff_val}</div><div class="sub">Jun 11, opener</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="wc-section">
          <h3>Title race</h3>
          <div class="wc-legend">
            <span><span class="wc-swatch" style="background:{GOLD};"></span>Top 8</span>
            <span><span class="wc-swatch" style="background:{GREEN};"></span>Chasing pack</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_n = 16
    shown = df.head(top_n)
    max_p = float(df["p"].max()) or 1.0
    rows = []
    for i, r in shown.iterrows():
        if i == 8:
            rows.append('<div class="wc-divider"><span>Chasing pack</span><span class="ln"></span></div>')
        gold = i < 8
        width = max(r["p"] / max_p * 100, 0.6)
        rows.append(
            f"""
            <div class="wc-row">
              <span class="wc-rank">{i + 1}</span>
              {flag_img(r['team'])}
              <span class="wc-team"><span class="nm">{html.escape(r['team'])}</span>
                <span class="cf">{confed(r['team'])}</span></span>
              <span class="wc-track"><span class="wc-fill"
                style="width:{width:.1f}%;background:{GOLD if gold else GREEN};"></span></span>
              <span class="wc-pct">{r['p'] * 100:.1f}%</span>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.markdown(
        f'<div class="wc-foot">Showing top {len(shown)} of {len(df)} teams · '
        f'generated {gen_full} · model retrains nightly.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 2 — Match Predictor
# ---------------------------------------------------------------------------

def render_match_predictor() -> None:
    """Team selectors with flags, a split probability bar, and outcome cards."""
    try:
        matches = fetch_match_predictions()
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
        return

    df = pd.DataFrame(matches)
    page_intro("Head to head",
               "Pick any group-stage pairing to see win, draw and loss probabilities.")

    col1, col2 = st.columns(2)
    home = col1.selectbox("Home team", sorted(df["home"].unique()))
    away_options = sorted(df.loc[df["home"] == home, "away"].unique())
    away = col2.selectbox("Away team", away_options)

    match = df[(df["home"] == home) & (df["away"] == away)].iloc[0]
    ph, pd_, pa = float(match["p_home_win"]), float(match["p_draw"]), float(match["p_away_win"])

    st.markdown(
        f'<div class="wc-context">{flag_img(home, 18)} <b>{html.escape(home)}</b> '
        f'&nbsp;vs&nbsp; {flag_img(away, 18)} <b>{html.escape(away)}</b></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="wc-split">
          <div style="width:{ph * 100:.1f}%;background:{GREEN};color:#10210A;">{ph * 100:.0f}%</div>
          <div style="width:{pd_ * 100:.1f}%;background:{GREY};color:#1A1C1E;">{pd_ * 100:.0f}%</div>
          <div style="width:{pa * 100:.1f}%;background:{GOLD};color:#3A2402;">{pa * 100:.0f}%</div>
        </div>
        <div class="wc-splitlbl"><span>{html.escape(home)} win</span><span>Draw</span>
          <span>{html.escape(away)} win</span></div>

        <div class="wc-cards">
          <div class="wc-card"><div class="hd">{flag_img(home)} {html.escape(home)} win</div>
            <div class="big" style="color:{GREEN_TXT};">{ph * 100:.1f}%</div>
            <div class="mini"><div style="width:{ph * 100:.0f}%;height:100%;background:{GREEN};"></div></div></div>
          <div class="wc-card"><div class="hd">Draw</div>
            <div class="big" style="color:#C7CCD1;">{pd_ * 100:.1f}%</div>
            <div class="mini"><div style="width:{pd_ * 100:.0f}%;height:100%;background:{GREY};"></div></div></div>
          <div class="wc-card"><div class="hd">{flag_img(away)} {html.escape(away)} win</div>
            <div class="big" style="color:{GOLD_TXT};">{pa * 100:.1f}%</div>
            <div class="mini"><div style="width:{pa * 100:.0f}%;height:100%;background:{GOLD};"></div></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    outcomes = {f"{home} win": ph, "Draw": pd_, f"{away} win": pa}
    best = max(outcomes, key=outcomes.get)
    st.markdown(
        f'<div class="wc-callout"><span style="font-size:18px;">🏆</span>'
        f'<span><b>Most likely result: {html.escape(best)} ({outcomes[best] * 100:.0f}%).</b> '
        f'Probabilities are calibrated from the latest model run.</span></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 3 — Odds Tracker
# ---------------------------------------------------------------------------

def _compute_movers(history: list[dict]) -> tuple[tuple, tuple] | None:
    """
    Return ((riser_team, delta_pts), (faller_team, delta_pts)) over the window,
    or None if there isn't enough history. Delta is in percentage points.
    """
    first, last = history[0]["tournament_odds"], history[-1]["tournament_odds"]
    deltas = {t: (last[t] - first.get(t, 0.0)) * 100 for t in last}
    if not deltas:
        return None
    riser = max(deltas.items(), key=lambda kv: kv[1])
    faller = min(deltas.items(), key=lambda kv: kv[1])
    return riser, faller


def render_odds_tracker() -> None:
    """Biggest-mover cards, flag chips, and a multi-line title-probability chart."""
    try:
        history = fetch_odds_history()
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
        return

    page_intro("How the odds are moving",
               "Title probability per team across nightly model re-runs.")

    if len(history) < 2:
        st.info("Not enough snapshots yet to chart a trend — check back after the next retrain.")
        return

    movers = _compute_movers(history)
    if movers:
        (rt, rd), (ft, fd) = movers
        st.markdown(
            f"""
            <div class="wc-kpis" style="grid-template-columns:repeat(2,1fr);">
              <div class="wc-mover"><div class="lbl">📈 Biggest riser · since first snapshot</div>
                <div class="bd">{flag_img(rt)} {html.escape(rt)}
                  <span style="margin-left:auto;color:{GREEN_TXT};">+{rd:.1f} pts</span></div></div>
              <div class="wc-mover"><div class="lbl">📉 Biggest faller · since first snapshot</div>
                <div class="bd">{flag_img(ft)} {html.escape(ft)}
                  <span style="margin-left:auto;color:{RED};">{fd:.1f} pts</span></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    long_rows = [
        {"generated_at": snap["generated_at"], "team": team, "p": prob}
        for snap in history
        for team, prob in snap["tournament_odds"].items()
    ]
    df = pd.DataFrame(long_rows)
    df["generated_at"] = pd.to_datetime(df["generated_at"], format="ISO8601")

    latest = history[-1]["tournament_odds"]
    default_teams = sorted(latest, key=latest.get, reverse=True)[:6]
    teams = st.multiselect("Teams to track", sorted(df["team"].unique()), default=default_teams)
    if not teams:
        st.info("Select at least one team to plot.")
        return

    chips = "".join(
        f'<span class="wc-chip">{flag_img(t, 18)} {html.escape(t)} '
        f'<span style="color:{TXT3};">{latest.get(t, 0) * 100:.1f}%</span></span>'
        for t in sorted(teams, key=lambda t: latest.get(t, 0), reverse=True)
    )
    st.markdown(f'<div class="wc-chips">{chips}</div>', unsafe_allow_html=True)

    axis_kw = dict(labelColor=TXT2, titleColor=TXT2, gridColor=CARD_BD,
                   domainColor=CARD_BD, tickColor=CARD_BD)
    chart = (
        alt.Chart(df[df["team"].isin(teams)])
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("generated_at:T", title="Snapshot", axis=alt.Axis(**axis_kw)),
            y=alt.Y("p:Q", title="Title probability",
                    axis=alt.Axis(format="%", **axis_kw)),
            color=alt.Color("team:N", title="Team",
                            scale=alt.Scale(scheme="tableau10"),
                            legend=alt.Legend(labelColor=TXT, titleColor=TXT2)),
            tooltip=[
                "team",
                alt.Tooltip("generated_at:T", title="Snapshot"),
                alt.Tooltip("p:Q", title="Title probability", format=".1%"),
            ],
        )
        .properties(height=480)
        .configure(background="transparent")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True, theme=None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PAGES = {
    "Tournament odds": render_tournament_odds,
    "Match predictor": render_match_predictor,
    "Odds tracker": render_odds_tracker,
}


def main() -> None:
    inject_theme()
    render_header()
    page = render_nav()
    PAGES[page]()


if __name__ == "__main__":
    main()
