"""
Streamlit dashboard for the WC 2026 Match Predictor — redesigned.

Pages (st.tabs):
  1. Tournament Odds  — Countdown banner · KPI strip · animated title-race bars.
  2. Match Predictor  — Team selectors · split bar · outcome cards.
  3. Standings & Bracket — 12-group grid ranked by title probability.
  4. Odds Tracker     — Biggest-mover cards · multi-line title-probability chart.

Theme: Barlow Condensed for display text · dark charcoal + pitch-green + gold.
All CSS lives in dashboard/theme.py; this file owns layout and data logic only.
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
import streamlit.components.v1 as components
from dotenv import load_dotenv
from theme import inject_theme

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CACHE_TTL_SECONDS = int(os.getenv("DASHBOARD_CACHE_TTL", "3600"))
REQUEST_TIMEOUT = 60

# Tournament calendar
KICKOFF_DATE = dt.date(2026, 6, 11)
KICKOFF_DT = dt.datetime(2026, 6, 11, 18, 0, 0, tzinfo=dt.timezone.utc)
TOURNAMENT_END = dt.date(2026, 7, 19)

# Color palette
BG       = "#1B1D1F"; CARD     = "#26292C"; CARD_BD  = "#34383C"; TRACK = "#303438"
TXT      = "#E8EAEC"; TXT2     = "#9AA0A6"; TXT3     = "#7A8087"
GREEN    = "#7FB83E"; GREEN_TXT = "#9FD45B"; GREEN_TINT = "#2A3A18"
GOLD     = "#F2A93B"; GOLD_TXT  = "#F2B85C"; GREY     = "#80868C"; RED = "#E5705B"
STATUS_OK = "#3BD68B"; STATUS_WARN = "#F2A93B"; STATUS_ERR = "#E5705B"

# Page icon
_ICON_PATH = Path(__file__).resolve().parent / "assets" / "ball_icon.svg"
PAGE_ICON  = str(_ICON_PATH) if _ICON_PATH.exists() else "⚽"

st.set_page_config(
    page_title="WC 2026 Match Predictor",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Team Reference Data
# ---------------------------------------------------------------------------

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
    "Czechia": "UEFA", "Switzerland": "UEFA", "Germany": "UEFA", "Netherlands": "UEFA",
    "Sweden": "UEFA", "Belgium": "UEFA", "Spain": "UEFA", "France": "UEFA",
    "Norway": "UEFA", "Austria": "UEFA", "Portugal": "UEFA", "England": "UEFA",
    "Croatia": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Brazil": "CONMEBOL", "Paraguay": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Argentina": "CONMEBOL", "Colombia": "CONMEBOL",
    "Mexico": "CONCACAF", "Canada": "CONCACAF", "Haiti": "CONCACAF",
    "USA": "CONCACAF", "Curaçao": "CONCACAF", "Panama": "CONCACAF",
    "South Africa": "CAF", "Morocco": "CAF", "Ivory Coast": "CAF", "Tunisia": "CAF",
    "Egypt": "CAF", "Cape Verde": "CAF", "Senegal": "CAF", "Algeria": "CAF",
    "Congo DR": "CAF", "Ghana": "CAF",
    "Korea Republic": "AFC", "Qatar": "AFC", "Australia": "AFC", "Japan": "AFC",
    "IR Iran": "AFC", "Saudi Arabia": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC", "New Zealand": "OFC",
}

# Primary and secondary hex colors drawn from each nation's flag
TEAM_COLORS: dict[str, tuple[str, str]] = {
    "Argentina":            ("#74ACDF", "#F6B40E"),
    "Algeria":              ("#006233", "#D21034"),
    "Australia":            ("#002B7F", "#FF0000"),
    "Austria":              ("#ED2939", "#FFD700"),
    "Belgium":              ("#FAE042", "#EF3340"),
    "Bosnia and Herzegovina": ("#002395", "#FCCA00"),
    "Brazil":               ("#009C3B", "#FFDF00"),
    "Canada":               ("#FF0000", "#A52A2A"),
    "Cape Verde":           ("#003893", "#CF2027"),
    "Colombia":             ("#FCD116", "#003087"),
    "Congo DR":             ("#007FFF", "#F7D618"),
    "Croatia":              ("#FF0000", "#003DA5"),
    "Czechia":              ("#D7141A", "#11457E"),
    "Curaçao":              ("#002B7F", "#F9E814"),
    "Ecuador":              ("#FFD100", "#0072CE"),
    "Egypt":                ("#CE1126", "#C09300"),
    "England":              ("#CF142B", "#012169"),
    "France":               ("#002395", "#ED2939"),
    "Germany":              ("#FFCE00", "#DD0000"),
    "Ghana":                ("#006B3F", "#FCD116"),
    "Haiti":                ("#00209F", "#D21034"),
    "IR Iran":              ("#239F40", "#DA0000"),
    "Iraq":                 ("#CE1126", "#007A3D"),
    "Ivory Coast":          ("#F77F00", "#009A44"),
    "Japan":                ("#BC002D", "#003087"),
    "Jordan":               ("#007A3D", "#CE1126"),
    "Korea Republic":       ("#003478", "#CD2E3A"),
    "Mexico":               ("#006847", "#CE1126"),
    "Morocco":              ("#C1272D", "#006233"),
    "Netherlands":          ("#FF6600", "#003082"),
    "New Zealand":          ("#CC0000", "#00247D"),
    "Norway":               ("#EF2B2D", "#003087"),
    "Panama":               ("#DB0000", "#0038A8"),
    "Paraguay":             ("#D52B1E", "#0038A8"),
    "Portugal":             ("#006600", "#FF0000"),
    "Qatar":                ("#8D1B3D", "#FFFFFF"),
    "Saudi Arabia":         ("#006C35", "#C8A951"),
    "Scotland":             ("#003078", "#C60C30"),
    "Senegal":              ("#00853F", "#FDEF42"),
    "South Africa":         ("#007A4D", "#FFB81C"),
    "Spain":                ("#C60B1E", "#FFC400"),
    "Sweden":               ("#006AA7", "#FECC02"),
    "Switzerland":          ("#FF0000", "#003082"),
    "Tunisia":              ("#E70013", "#C09300"),
    "Turkey":               ("#E30A17", "#C09300"),
    "Uruguay":              ("#75AADB", "#FFFFFF"),
    "USA":                  ("#B22234", "#002868"),
    "Uzbekistan":           ("#1EB53A", "#009FCA"),
}

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# ---------------------------------------------------------------------------
# HTML Helpers
# ---------------------------------------------------------------------------

def ball_svg(size: int = 22, color: str | None = None) -> str:
    """Inline soccer-ball brand mark as a self-contained SVG string.

    Defaults to the pitch-green gradient; pass a hex `color` for a flat tint.
    """
    if color:
        stroke = f'stroke="{color}"'
        defs = ""
    else:
        stroke = 'stroke="url(#bG)"'
        defs = (
            '<defs><linearGradient id="bG" gradientUnits="userSpaceOnUse" x1="291" y1="32" x2="291" y2="550">'
            '<stop offset="0" stop-color="#9EDB35"/><stop offset="1" stop-color="#5DA10C"/>'
            '</linearGradient></defs>'
        )
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 582 582" '
        'xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;flex:none;">'
        f'{defs}'
        f'<circle cx="291" cy="291" r="259" fill="none" {stroke} stroke-width="63"/>'
        f'<path d="M291,148.5 L426.5,247 L374.8,406.3 L207.2,406.3 L155.5,247 Z" '
        f'fill="none" {stroke} stroke-width="63" stroke-linejoin="round"/>'
        f'<g {stroke} stroke-width="63">'
        '<line x1="291" y1="148.5" x2="291" y2="32"/>'
        '<line x1="426.5" y1="247" x2="537.3" y2="211"/>'
        '<line x1="374.8" y1="406.3" x2="443.2" y2="500.5"/>'
        '<line x1="207.2" y1="406.3" x2="138.8" y2="500.5"/>'
        '<line x1="155.5" y1="247" x2="44.7" y2="211"/>'
        '</g></svg>'
    )


def trend_up_svg(size: int = 16) -> str:
    """Upward-trending line chart icon in pitch-green."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 16 16" '
        'xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;flex:none;">'
        '<polyline points="1,13 5,8 9,10 15,3" fill="none" stroke="#9FD45B" '
        'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        '<polyline points="11,3 15,3 15,7" fill="none" stroke="#9FD45B" '
        'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        '</svg>'
    )


def trend_down_svg(size: int = 16) -> str:
    """Downward-trending line chart icon in red."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 16 16" '
        'xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;flex:none;">'
        '<polyline points="1,3 5,8 9,6 15,13" fill="none" stroke="#E5705B" '
        'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        '<polyline points="11,13 15,13 15,9" fill="none" stroke="#E5705B" '
        'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        '</svg>'
    )


def flag_img(team: str, height: int = 22) -> str:
    """Return an "img" tag for a team's flag, or a blank placeholder if unknown."""
    iso = TEAM_ISO.get(team)
    width = round(height * 4 / 3)
    style = (
        f"width:{width}px;height:{height}px;border-radius:3px;object-fit:cover;"
        "border:0.5px solid rgba(255,255,255,0.22);vertical-align:middle;flex:none;"
    )
    if not iso:
        return f'<span style="{style}background:#3A3E42;display:inline-block;"></span>'
    return f'<img src="https://flagcdn.com/h40/{iso}.png" alt="" style="{style}">'


def confed(team: str) -> str:
    """Return a team's confederation acronym, or an empty string if unknown."""
    return TEAM_CONFED.get(team, "")


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

def _get(endpoint: str) -> dict | list:
    """
    Fetch JSON data from the FastAPI backend.
    
    Constructs the full API URL, sends a GET request with a timeout, validates the HTTP
    response, and returns the parsed JSON. Raises and exception fi the request times
    out or the API returns a 4xx/5xx status.
    
    Args:
        endpoint: The API path relative to `API_BASE_URL` (e.g. "/predictions/matches")
        
    Returns:
        dict | list: Parsed JSON response from the API.
    """
    r = requests.get(f"{API_BASE_URL}{endpoint}", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_tournament_odds() -> dict:
    """Cached wrapper around GET /predictions/tournament."""
    return _get("/predictions/tournament")


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_match_predictions() -> list[dict]:
    """Cached wrapper around GET /predictions/matches."""
    return _get("/predictions/matches")


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_odds_history() -> list[dict]:
    """Cached wrapper around GET /predictions/history."""
    return _get("/predictions/history")


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_knockout_bracket() -> dict:
    """Cached wrapper around GET /predictions/knockout."""
    return _get("/predictions/knockout")


@st.cache_data(ttl=60)
def fetch_health() -> dict:
    """Short-TTL wrapper around GET /health so the status dot stays fresh."""
    return _get("/health")


def _format_generated(raw: str | None) -> str:
    """Render an ISO timestamp as `Jun 7, 2026 20:33 UTC`."""
    if not raw:
        return "unknown"
    try:
        ts = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return f"{ts:%b} {ts.day}, {ts:%Y %H:%M} UTC"
    except (ValueError, TypeError):
        return raw


# ---------------------------------------------------------------------------
# Shared Chrome
# ---------------------------------------------------------------------------

def render_header() -> None:
    """    
    Display the dashboard header with an API health status.
    
    Fetches the API's health endpoint to determine operational status and renders a fixed
    header row with the WC 2026 branding, a colored status dot, and a human-readable status
    label. When available, also displays the timestamp of the latest prediction generation.
    
    Args:
        None:
        
    Returns:
        None:
    """
    dot, label, ts = STATUS_WARN, "API status unknown", ""
    
    try:
        health = fetch_health()
        # Check if API is fully healthy
        if (
            health.get("status") == "ok"
            and health.get("model_loaded")
            and health.get("predictions_loaded")):
            dot, label = STATUS_OK, "API Live"
        else:
            dot, label = STATUS_WARN, "API Degraded"
        # Extract "generated_at" timestamp
        ts = f" · Predictions {_format_generated(health.get('generated_at'))}"
        
    except requests.RequestException:
        dot, label = STATUS_ERR, "API Unreachable"

    # Render custom HTML header row
    ts_html = f'<span class="wc-api-ts">{html.escape(ts)}</span>' if ts else ""
    st.markdown(
        f'<div class="wc-header">'
        f'  <div class="wc-brand">{ball_svg(22)} WC 2026 Predictor</div>'
        f'  <div class="wc-api-status">'
        f'    <span class="wc-dot" style="background:{dot};"></span>'
        f'    {html.escape(label)}{ts_html}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _banner_component_html(ticker: str, tournament_status: str = "UNDERWAY") -> str:
    """Self-contained HTML for the countdown banner component — ticks every second via JS."""
    return (
        "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
        "<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
        "<link href='https://fonts.googleapis.com/css2?family=Barlow+Condensed"
        ":wght@500;700;800&family=Barlow:wght@400;500&display=swap' rel='stylesheet'>"
        "<style>"
        "*{box-sizing:border-box;margin:0;padding:0}"
        "body{background:#26292C;overflow:hidden;font-family:'Barlow',sans-serif;color:#E8EAEC}"
        ".wc-banner{background:#26292C;position:relative;overflow:hidden}"
        ".wc-banner::before{content:'';position:absolute;inset:0;pointer-events:none;"
        "background:repeating-linear-gradient(180deg,"
        "rgba(127,184,62,.055) 0,rgba(127,184,62,.055) 28px,"
        "rgba(127,184,62,.018) 28px,rgba(127,184,62,.018) 56px)}"
        ".wc-banner-inner{display:grid;grid-template-columns:auto 1fr auto;"
        "align-items:center;gap:40px;padding:20px 24px;position:relative;z-index:1}"
        ".wc-tourn-lbl{font-size:11px;font-weight:700;letter-spacing:3px;"
        "text-transform:uppercase;color:#7A8087;margin-bottom:4px}"
        ".wc-tourn-name{font-family:'Barlow Condensed',sans-serif;font-size:30px;"
        "font-weight:800;letter-spacing:1px;line-height:1;color:#E8EAEC}"
        ".wc-tourn-dates{font-size:13px;color:#7A8087;margin-top:5px}"
        ".wc-countdown{display:flex;align-items:flex-end;gap:2px;justify-content:center}"
        ".wc-cd-unit{text-align:center;padding:0 6px}"
        ".wc-cd-val{display:block;font-family:'Barlow Condensed',sans-serif;font-size:52px;"
        "font-weight:800;color:#9FD45B;line-height:1}"
        ".wc-cd-lbl{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;"
        "color:#7A8087;display:block;margin-top:3px}"
        ".wc-cd-sep{font-family:'Barlow Condensed',sans-serif;font-size:40px;font-weight:800;"
        "color:#3A5A22;margin-bottom:14px;line-height:1}"
        ".wc-ticker-outer{border-left:1px solid #34383C;padding-left:24px;overflow:hidden}"
        ".wc-ticker-lbl{font-size:10px;font-weight:700;letter-spacing:2.5px;"
        "text-transform:uppercase;color:#7FB83E;margin-bottom:7px}"
        ".wc-ticker-mask{overflow:hidden}"
        ".wc-ticker-track{display:flex;gap:40px;white-space:nowrap;"
        "animation:wc-ticker 42s linear infinite;width:max-content}"
        ".wc-ticker-item{display:inline-flex;align-items:center;gap:8px;"
        "font-size:14px;color:#9AA0A6;flex-shrink:0}"
        ".wc-ticker-item strong{color:#E8EAEC;font-weight:600}"
        ".wc-tick-val{font-family:'Barlow Condensed',sans-serif;font-size:17px;"
        "font-weight:700;color:#9FD45B}"
        "@keyframes wc-ticker{from{transform:translateX(0)}to{transform:translateX(-50%)}}"
        "@media (max-width:600px){"
        ".wc-banner-inner{grid-template-columns:1fr;gap:14px;padding:16px}"
        ".wc-ticker-outer{display:none}"
        ".wc-tourn-name{font-size:22px}"
        ".wc-cd-val{font-size:38px}"
        ".wc-cd-sep{font-size:28px;margin-bottom:10px}"
        "}"
        "</style></head><body>"
        "<div class='wc-banner'><div class='wc-banner-inner'>"
        "<div>"
        "<div class='wc-tourn-lbl'>FIFA World Cup</div>"
        "<div class='wc-tourn-name'>USA &middot; CANADA &middot; MEXICO</div>"
        "<div class='wc-tourn-dates'>Jun 11 &ndash; Jul 19, 2026</div>"
        "</div>"
        "<div id='wc-cd'></div>"
        "<div class='wc-ticker-outer'>"
        "<div class='wc-ticker-lbl'>Top title odds</div>"
        "<div class='wc-ticker-mask'>"
        f"<div class='wc-ticker-track'>{ticker}</div>"
        "</div></div>"
        "</div></div>"
        "<script>"
        "var T=new Date('2026-06-11T18:00:00Z').getTime();"
        "var el=document.getElementById('wc-cd');"
        "function p(n){return String(n).padStart(2,'0');}"
        "function tick(){"
        "var delta=T-Date.now();"
        "if(delta<=0){"
        f"el.innerHTML='<div class=\"wc-countdown\" style=\"font-family:Barlow Condensed,sans-serif;"
        f"font-size:40px;font-weight:800;color:#9FD45B;letter-spacing:2px;\">{tournament_status}</div>';"
        "return;}"
        "var d=Math.floor(delta/86400000);"
        "var h=Math.floor((delta%86400000)/3600000);"
        "var m=Math.floor((delta%3600000)/60000);"
        "var s=Math.floor((delta%60000)/1000);"
        "el.innerHTML="
        "'<div class=\"wc-countdown\">'"
        "+'<div class=\"wc-cd-unit\"><span class=\"wc-cd-val\">'+p(d)+'</span>"
        "<span class=\"wc-cd-lbl\">Days</span></div>'"
        "+'<span class=\"wc-cd-sep\">:</span>'"
        "+'<div class=\"wc-cd-unit\"><span class=\"wc-cd-val\">'+p(h)+'</span>"
        "<span class=\"wc-cd-lbl\">Hrs</span></div>'"
        "+'<span class=\"wc-cd-sep\">:</span>'"
        "+'<div class=\"wc-cd-unit\"><span class=\"wc-cd-val\">'+p(m)+'</span>"
        "<span class=\"wc-cd-lbl\">Min</span></div>'"
        "+'<span class=\"wc-cd-sep\">:</span>'"
        "+'<div class=\"wc-cd-unit\"><span class=\"wc-cd-val\">'+p(s)+'</span>"
        "<span class=\"wc-cd-lbl\">Sec</span></div>'"
        "+'</div>';}"
        "tick();setInterval(tick,1000);"
        "function fit(){if(window.frameElement){"
        "window.frameElement.style.height=(document.body.scrollHeight+2)+'px';}}"
        "window.addEventListener('load',fit);"
        "window.addEventListener('resize',fit);"
        "if(document.fonts&&document.fonts.ready){document.fonts.ready.then(fit);}"
        "setTimeout(fit,300);setTimeout(fit,1200);fit();"
        "</script></body></html>"
    )


def render_banner() -> None:
    """
    Display a live countdown banner with top-8 tournament favorites and animated ticker.

    Fetches tournament odds from the API and renders an animated banner featuring: (1) a
    countdown timer ticking down to World Cup kickoff, (2) a seamlessly scrolling ticker
    of the top 8 teams by title probability.

    Args:
        None:

    Returns:
        None:
    """
    # Attempt to fetch tournament odds
    try:
        payload = fetch_tournament_odds()
        odds: dict[str, float] = payload["tournament_odds"]
    except requests.RequestException:
        odds = {}

    # Extract top 8 teams
    top8 = sorted(odds.items(), key=lambda kv: kv[1], reverse=True)[:8]

    # Construct an HTML string with all 8 teams
    ticker = "".join(
        f'<span class="wc-ticker-item">{flag_img(t, 16)}'
        f' <strong>{html.escape(t)}</strong>'
        f' <span class="wc-tick-val">{p * 100:.1f}%</span></span>'
        for t, p in top8
    ) * 2

    # Determine tournament status
    today = dt.date.today()
    tournament_status = "COMPLETE" if today > TOURNAMENT_END else "UNDERWAY"

    # Wrap ticker in self-contained HTML page
    components.html(_banner_component_html(ticker, tournament_status), height=120, scrolling=False)


# ---------------------------------------------------------------------------
# Page 1 — Tournament Odds
# ---------------------------------------------------------------------------

def render_tournament_odds() -> None:
    """
    Render the tournament odds page: KPI strip, ranked bars, and team breakdown.
    
    Fetches the latest tournament probabilities from the API and displays them as a
    professional data-journalism-style page. Defaults to showing the top 16 teams, however,
    users can toggle to view all 48 teams.
    
    Args:
        None:
        
    Returns:
        None:
    """
    # Attempt to fetch tournament odds
    try:
        payload = fetch_tournament_odds()
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
        return

    # Extract odds into a DataFrame
    odds = payload["tournament_odds"]
    df = (
        pd.DataFrame(odds.items(), columns=["team", "p"])
        .sort_values("p", ascending=False)
        .reset_index(drop=True)
    )
    n_sims = payload.get("n_simulations", 0)
    gen_full = _format_generated(payload.get("generated_at"))

    st.markdown(
        f'<p class="wc-eyebrow">Each team\'s probability of lifting the trophy, '
        f'across {n_sims:,} simulated tournaments.</p>',
        unsafe_allow_html=True,
    )

    fav = df.iloc[0]
    today = dt.date.today()
    days_left = (KICKOFF_DATE - today).days

    if today > TOURNAMENT_END:
        ko_val = "Complete"
    elif days_left < 0:
        ko_val = "Underway"
    elif days_left == 0:
        ko_val = "Today"
    else:
        ko_val = f"{days_left} days"

    # 4-column KPI strip
    st.markdown(
        f"""
        <div class="wc-kpis">
          <div class="wc-kpi">
            <div class="lbl">Favourite</div>
            <div class="val" style="font-size:36px;">{flag_img(fav["team"])} {fav["p"]*100:.1f}%</div>
            <div class="sub">{html.escape(fav["team"])}</div>
          </div>
          <div class="wc-kpi">
            <div class="lbl">Simulations</div>
            <div class="val">{n_sims:,}</div>
            <div class="sub">Monte Carlo Runs</div>
          </div>
          <div class="wc-kpi">
            <div class="lbl">Field</div>
            <div class="val">{len(df)}</div>
            <div class="sub">Teams Qualified</div>
          </div>
          <div class="wc-kpi">
            <div class="lbl">Kick-off</div>
            <div class="val" style="font-size:36px;">{html.escape(ko_val)}</div>
            <div class="sub">Jun 11, Opener</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pitch centre-circle divider
    st.markdown(
        """
        <div class="wc-pitch-div">
          <svg width="100%" height="24" xmlns="http://www.w3.org/2000/svg">
            <line x1="0" y1="12" x2="48%" y2="12" stroke="#3A5A22" stroke-width="1.5"/>
            <circle cx="50%" cy="12" r="10" fill="none" stroke="#3A5A22" stroke-width="1.5"/>
            <line x1="52%" y1="12" x2="100%" y2="12" stroke="#3A5A22" stroke-width="1.5"/>
          </svg>
        </div>
        <div class="wc-section">
          <h3>Title Race</h3>
          <div class="wc-legend">
            <span><span class="wc-swatch" style="background:#F2A93B;"></span>Top 8</span>
            <span><span class="wc-swatch" style="background:#7FB83E;"></span>Chasing pack</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "show_all_odds" not in st.session_state:
        st.session_state.show_all_odds = False

    shown = df if st.session_state.show_all_odds else df.head(16)
    max_p = float(df["p"].max()) or 1.0
    rows: list[str] = []
    
    # Loop through each team
    for i, r in shown.iterrows():
        if i == 8:
            rows.append(
                '<div class="wc-divider">'
                '<span class="ln"></span><span>Chasing Pack</span><span class="ln"></span>'
                '</div>'
            )
        elif i == 16 and st.session_state.show_all_odds:
            rows.append(
                '<div class="wc-divider">'
                '<span class="ln"></span><span>Rest of the Field</span><span class="ln"></span>'
                '</div>'
            )
        gold = i < 8
        width = max(r["p"] / max_p * 100, 0.5)
        rows.append(
            f'<div class="wc-row">'
            f'  <span class="wc-rank">{i + 1}</span>'
            f'  {flag_img(r["team"])}'
            f'  <span class="wc-team">'
            f'    <span class="nm">{html.escape(r["team"])}</span>'
            f'    <span class="cf">{confed(r["team"])}</span>'
            f'  </span>'
            f'  <span class="wc-track">'
            f'    <span class="wc-fill" style="width:{width:.1f}%;background:{GOLD if gold else GREEN};"></span>'
            f'  </span>'
            f'  <span class="wc-pct">{r["p"] * 100:.1f}%</span>'
            f'</div>'
        )

    st.markdown("".join(rows), unsafe_allow_html=True)

    # Create "Show all" button and functionality
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        btn_label = "Show less" if st.session_state.show_all_odds else f"View all {len(df)}"
        if st.button(btn_label, key="toggle_all_odds", use_container_width=True):
            st.session_state.show_all_odds = not st.session_state.show_all_odds
            st.rerun()

    # Render footer
    showing_label = "all" if st.session_state.show_all_odds else f"top {len(shown)}"
    st.markdown(
        f'<div class="wc-foot">Showing {showing_label} of {len(df)} teams &middot; '
        f'Generated {gen_full} &middot; Model retrains nightly.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 2 — Match Predictor (shared knockout helpers)
# ---------------------------------------------------------------------------

_MAX_NM = 14


def _short(name: str | None) -> str:
    """Truncate a team name to _MAX_NM chars to prevent overflow in card layouts."""
    if not name:
        return ""
    return name if len(name) <= _MAX_NM else name[:_MAX_NM].rstrip() + "…"


_KO_LABEL = {
    "R32": "Round of 32", "R16": "Round of 16",
    "QF": "Quarter Finals", "SF": "Semi Finals", "Final": "Final",
}
_KO_COLS = {"R32": 4, "R16": 4, "QF": 4, "SF": 2, "Final": 1}


def _ko_team_label(name: str | None) -> str:
    """Flag + name for a knockout participant, or a 'TBD' placeholder if undetermined."""
    if not name:
        return f'{flag_img("", 18)} <span class="wc-ko-nm wc-ko-tbd">TBD</span>'
    return f'{flag_img(name, 18)} <span class="wc-ko-nm" title="{html.escape(name)}">{html.escape(_short(name))}</span>'


def _render_ko_match_card(match: dict) -> str:
    """
    Return HTML for a single knockout match card.

    Played matches show the final score (with penalties when decided on spot-kicks) and a
    winner badge; upcoming matches show the model win/loss split bar and the predicted team
    expected to advance.
    """
    home, away = match.get("home"), match.get("away")
    winner = match.get("winner")
    teams_row = (
        f'<div class="wc-ko-teams">'
        f'{_ko_team_label(home)}'
        f'<span class="wc-ko-vs">vs</span>'
        f'{_ko_team_label(away)}'
        f'</div>'
    )

    if match.get("status") == "played":
        hs, as_ = match.get("home_score"), match.get("away_score")
        hp, ap = match.get("home_pens"), match.get("away_pens")
        pens = (f' <span class="wc-ko-pens">({hp}&ndash;{ap} pens)</span>'
                if hp is not None and ap is not None else "")
        score_html = (
            f'<div class="wc-ko-score">'
            f'<span class="t">{html.escape(_short(home) or "TBD")}</span>'
            f'<span class="sc">{hs}&ndash;{as_}</span>'
            f'<span class="t">{html.escape(_short(away) or "TBD")}</span>'
            f'{pens}'
            f'</div>'
        )
        badge = (
            f'<div class="wc-ko-winner wc-ko-winner-played">{ball_svg(14, color=GOLD)}'
            f'<span><strong>{html.escape(_short(winner))}</strong> advance</span></div>'
            if winner else ""
        )
        return f'<div class="wc-ko-card wc-ko-played">{teams_row}{score_html}{badge}</div>'

    # Scheduled / upcoming — model probability split bar + predicted advancer.
    ph, pa = match["p_home_win"], match["p_away_win"]
    predicted = winner or (home if ph >= pa else away) or "TBD"
    return (
        f'<div class="wc-ko-card">'
        f'{teams_row}'
        f'<div class="wc-split">'
        f'<div style="width:{ph*100:.1f}%;background:{GREEN};color:#0d1f06;">{ph*100:.0f}%</div>'
        f'<div style="width:{pa*100:.1f}%;background:{GOLD};color:#2a1500;">{pa*100:.0f}%</div>'
        f'</div>'
        f'<div class="wc-splitlbl">'
        f'<span>{html.escape(_short(home) or "TBD")}</span><span>{html.escape(_short(away) or "TBD")}</span>'
        f'</div>'
        f'<div class="wc-ko-adv">{ball_svg(14)}'
        f'<span>Expected to advance: <strong>{html.escape(_short(predicted))}</strong></span></div>'
        f'</div>'
    )


def _render_ko_round(round_name: str, matches: list[dict]) -> None:
    """Render all matches for one knockout round as a responsive card grid."""
    label = _KO_LABEL.get(round_name, round_name)
    n_cols = _KO_COLS.get(round_name, 4)
    st.markdown(
        f'<p class="wc-eyebrow">{label} — real results where matches are played, '
        f'model predictions for upcoming and future-round matchups.</p>',
        unsafe_allow_html=True,
    )
    cards_html = "".join(_render_ko_match_card(m) for m in matches)
    st.markdown(
        f'<div class="wc-ko-grid wc-ko-cols-{n_cols}">{cards_html}</div>',
        unsafe_allow_html=True,
    )


def render_match_predictor() -> None:
    """Group-stage team selectors + 5 knockout-stage subtabs."""
    gs_tab, r32_tab, r16_tab, qf_tab, sf_tab, final_tab = st.tabs(
        ["Group Stage", "Round of 32", "Round of 16", "Quarter Finals", "Semi Finals", "Final"]
    )

    # --- Group Stage ---
    with gs_tab:
        try:
            matches = fetch_match_predictions()
        except requests.RequestException as exc:
            st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
            return

        df = pd.DataFrame(matches)
        st.markdown(
            '<p class="wc-eyebrow">Pick any group-stage pairing to see win, draw and loss probabilities.</p>',
            unsafe_allow_html=True,
        )

        teams = sorted(set(df["home"]) | set(df["away"]))

        col1, col2 = st.columns(2)
        home = col1.selectbox("Home team", teams)

        opponents = sorted(
            set(df.loc[df["home"] == home, "away"])
            | set(df.loc[df["away"] == home, "home"])
        )
        away = col2.selectbox("Away team", opponents)

        fixture = df[
            ((df["home"] == home) & (df["away"] == away))
            | ((df["home"] == away) & (df["away"] == home))
        ].iloc[0]

        if fixture["home"] == home:
            ph, pd_, pa = float(fixture["p_home_win"]), float(fixture["p_draw"]), float(fixture["p_away_win"])
        else:
            ph, pd_, pa = float(fixture["p_away_win"]), float(fixture["p_draw"]), float(fixture["p_home_win"])

        st.markdown(
            f'<div class="wc-context">'
            f'  {flag_img(home, 18)} <strong>{html.escape(home)}</strong>'
            f'  &nbsp;vs&nbsp;'
            f'  {flag_img(away, 18)} <strong>{html.escape(away)}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="wc-split">
              <div style="width:{ph*100:.1f}%;background:{GREEN};color:#0d1f06;">{ph*100:.0f}%</div>
              <div style="width:{pd_*100:.1f}%;background:{GREY};color:#111;">{pd_*100:.0f}%</div>
              <div style="width:{pa*100:.1f}%;background:{GOLD};color:#2a1500;">{pa*100:.0f}%</div>
            </div>
            <div class="wc-splitlbl">
              <span>{html.escape(home)} win</span><span>Draw</span><span>{html.escape(away)} win</span>
            </div>
            <div class="wc-cards">
              <div class="wc-card">
                <div class="hd">{flag_img(home)} {html.escape(home)} win</div>
                <div class="big" style="color:{GREEN_TXT};">{ph*100:.1f}%</div>
                <div class="mini"><div style="width:{ph*100:.0f}%;height:100%;background:{GREEN};"></div></div>
              </div>
              <div class="wc-card">
                <div class="hd">Draw</div>
                <div class="big" style="color:#C7CCD1;">{pd_*100:.1f}%</div>
                <div class="mini"><div style="width:{pd_*100:.0f}%;height:100%;background:{GREY};"></div></div>
              </div>
              <div class="wc-card">
                <div class="hd">{flag_img(away)} {html.escape(away)} win</div>
                <div class="big" style="color:{GOLD_TXT};">{pa*100:.1f}%</div>
                <div class="mini"><div style="width:{pa*100:.0f}%;height:100%;background:{GOLD};"></div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        outcomes = {f"{home} win": ph, "Draw": pd_, f"{away} win": pa}
        best = max(outcomes, key=outcomes.get)
        st.markdown(
            f'<div class="wc-callout">{ball_svg(20)}'
            f'<span><strong>Most likely: {html.escape(best)} ({outcomes[best]*100:.0f}%)</strong>'
            f' &mdash; Calibrated from the latest model run.</span></div>',
            unsafe_allow_html=True,
        )

    # --- Knockout stages (shared fetch) ---
    try:
        ko_payload = fetch_knockout_bracket()
        bracket = ko_payload["knockout_bracket"]
    except requests.RequestException as exc:
        for tab in (r32_tab, r16_tab, qf_tab, sf_tab, final_tab):
            with tab:
                st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
        return

    with r32_tab:
        _render_ko_round("R32", bracket["R32"])
    with r16_tab:
        _render_ko_round("R16", bracket["R16"])
    with qf_tab:
        _render_ko_round("QF", bracket["QF"])
    with sf_tab:
        _render_ko_round("SF", bracket["SF"])
    with final_tab:
        _render_ko_round("Final", bracket["Final"])


# ---------------------------------------------------------------------------
# Page 3 — Standings & Bracket (bracket-tree helpers live here too)
# ---------------------------------------------------------------------------

def _bracket_connector_svg(slot_h: int, n_source: int, width: int = 24) -> str:
    """SVG connector column bridging n_source slots in round R to n_source//2 slots in round R+1."""
    n_groups = n_source // 2
    total_h  = slot_h * n_source
    x_bridge = width // 2

    lines: list[str] = []
    for i in range(n_groups):
        y_a   = i * 2 * slot_h + slot_h // 2           # center of first source slot
        y_b   = i * 2 * slot_h + slot_h + slot_h // 2  # center of second source slot
        y_mid = i * 2 * slot_h + slot_h                 # midpoint → target slot center
        c = "#34383C"
        lines += [
            f'<line x1="0" y1="{y_a}" x2="{x_bridge}" y2="{y_a}" stroke="{c}" stroke-width="1.5"/>',
            f'<line x1="0" y1="{y_b}" x2="{x_bridge}" y2="{y_b}" stroke="{c}" stroke-width="1.5"/>',
            f'<line x1="{x_bridge}" y1="{y_a}" x2="{x_bridge}" y2="{y_b}" stroke="{c}" stroke-width="1.5"/>',
            f'<line x1="{x_bridge}" y1="{y_mid}" x2="{width}" y2="{y_mid}" stroke="{c}" stroke-width="1.5"/>',
        ]
    return (
        f'<svg width="{width}" height="{total_h}" xmlns="http://www.w3.org/2000/svg"'
        f' style="flex-shrink:0;display:block;">'
        + "".join(lines)
        + "</svg>"
    )


def _bt_match_card(match: dict) -> str:
    """HTML for one bracket matchup card: two team rows (played score or scheduled probability)."""
    home   = match.get("home")
    away   = match.get("away")
    played = match.get("status") == "played"
    winner = match.get("winner")
    ph     = match.get("p_home_win") or 0.5
    pa     = match.get("p_away_win") or 0.5

    def _flag(team: str | None) -> str:
        iso = TEAM_ISO.get(team or "", "")
        if iso:
            return (
                f'<img src="https://flagcdn.com/h40/{iso}.png" width="22" height="16"'
                ' style="border-radius:2px;object-fit:cover;'
                'border:0.5px solid rgba(255,255,255,0.15);flex-shrink:0;" alt="">'
            )
        return (
            '<span style="display:inline-block;width:22px;height:16px;'
            'background:#3A3E42;border-radius:2px;flex-shrink:0;"></span>'
        )

    def _row(name: str | None, is_win: bool, score: int | None, prob: float | None) -> str:
        base = (
            "display:flex;align-items:center;gap:6px;padding:5px 9px;"
            "font-size:14px;min-height:28px;"
        )
        if name is None:
            return (
                f'<div style="{base}color:#7A8087;">'
                f'{_flag(None)}'
                f'<span style="flex:1;font-style:italic;">TBD</span>'
                f'<span style="min-width:28px;text-align:right;">—</span>'
                f'</div>'
            )
        nm = html.escape(name)
        if played:
            bg   = "rgba(63,92,31,0.35)" if is_win else "transparent"
            nc   = "#9FD45B"  if is_win else "#7A8087"
            sc_c = "#E8EAEC" if is_win else "#7A8087"
            val  = str(score) if score is not None else "—"
            return (
                f'<div style="{base}background:{bg};">'
                f'{_flag(name)}'
                f'<span style="flex:1;overflow:hidden;text-overflow:ellipsis;'
                f'white-space:nowrap;color:{nc};">{nm}</span>'
                f'<span style="font-family:\'Barlow Condensed\',sans-serif;font-weight:700;'
                f'font-size:16px;color:{sc_c};min-width:18px;text-align:right;">{val}</span>'
                f'</div>'
            )
        else:
            pc  = "#9FD45B" if (prob is not None and prob >= 0.5) else "#9AA0A6"
            val = f"{prob * 100:.0f}%" if prob is not None else "—"
            return (
                f'<div style="{base}color:#9AA0A6;">'
                f'{_flag(name)}'
                f'<span style="flex:1;overflow:hidden;text-overflow:ellipsis;'
                f'white-space:nowrap;">{nm}</span>'
                f'<span style="font-family:\'Barlow Condensed\',sans-serif;font-size:14px;'
                f'color:{pc};min-width:28px;text-align:right;">{val}</span>'
                f'</div>'
            )

    home_row = _row(home, winner == home if played else False,
                    match.get("home_score") if played else None,
                    ph if not played else None)
    away_row = _row(away, winner == away if played else False,
                    match.get("away_score") if played else None,
                    pa if not played else None)

    pens = ""
    if played and match.get("home_pens") is not None:
        pens = (
            f'<div style="font-size:11px;color:#7A8087;text-align:center;padding:3px 6px;">'
            f'({match["home_pens"]}&ndash;{match["away_pens"]} pens)</div>'
        )

    border = (
        "border:1px solid #3A5A22;border-left:3px solid #7FB83E;"
        if played else "border:1px solid #34383C;"
    )
    return (
        f'<div style="background:#26292C;{border}border-radius:8px;'
        f'width:200px;overflow:hidden;flex-shrink:0;">'
        f'{home_row}'
        f'<div style="height:1px;background:rgba(52,56,60,0.6);"></div>'
        f'{away_row}'
        f'{pens}'
        f'</div>'
    )


def _build_bracket_tree_html(bracket: dict) -> str:
    """Self-contained HTML page for the traditional left-to-right knockout bracket tree."""
    SLOT       = 90   # R32 slot height (px); doubles each successive round
    C_W        = 200  # card width (px)
    CN_W       = 32   # connector SVG width (px)
    ROUNDS     = ["R32", "R16", "QF", "SF", "Final"]
    RND_LABELS = {
        "R32": "Round of 32", "R16": "Round of 16",
        "QF": "Quarter Finals", "SF": "Semi Finals", "Final": "Final",
    }
    RND_COUNTS = {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "Final": 1}

    lbl_sty = (
        f"width:{C_W}px;text-align:center;font-size:11px;font-weight:700;"
        "letter-spacing:1.5px;text-transform:uppercase;color:#7A8087;"
        "padding-bottom:10px;flex-shrink:0;"
    )
    hdrs: list[str] = []
    for i, rnd in enumerate(ROUNDS):
        hdrs.append(f'<div style="{lbl_sty}">{RND_LABELS[rnd]}</div>')
        if i < len(ROUNDS) - 1:
            hdrs.append(f'<div style="width:{CN_W}px;flex-shrink:0;"></div>')
    header_html = (
        f'<div style="display:flex;min-width:max-content;">{"".join(hdrs)}</div>'
    )

    col_parts: list[str] = []
    for round_idx, rnd in enumerate(ROUNDS):
        slot_h  = SLOT * (2 ** round_idx)
        n_exp   = RND_COUNTS[rnd]
        matches = list(bracket.get(rnd, []))
        while len(matches) < n_exp:     # pad with empty placeholders if data is incomplete
            matches.append({})

        slots = [
            f'<div style="height:{slot_h}px;display:flex;align-items:center;flex-shrink:0;">'
            f'{_bt_match_card(m)}</div>'
            for m in matches
        ]
        col_parts.append(
            f'<div style="display:flex;flex-direction:column;flex-shrink:0;">'
            f'{"".join(slots)}</div>'
        )
        if round_idx < len(ROUNDS) - 1:
            col_parts.append(_bracket_connector_svg(slot_h, n_exp, CN_W))

    bracket_html = (
        f'<div style="display:flex;align-items:flex-start;min-width:max-content;">'
        f'{"".join(col_parts)}</div>'
    )

    font = (
        "<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
        "<link href='https://fonts.googleapis.com/css2?family=Barlow+Condensed"
        ":wght@700&family=Barlow:wght@400;500&display=swap' rel='stylesheet'>"
    )
    css = (
        "*{box-sizing:border-box;margin:0;padding:0}"
        "body{background:#1B1D1F;font-family:'Barlow',sans-serif;overflow:hidden;}"
        "#sw{width:100%;padding:16px;overflow-x:auto;-webkit-overflow-scrolling:touch;"
        "box-sizing:border-box;}"
    )
    natural = 5 * C_W + 4 * CN_W
    js = (
        "(function(){"
        "var o=document.getElementById('bk-outer');"
        "var w=document.getElementById('sw');"
        f"var n={natural};"
        "var m=0.75;"
        "function fit(){"
        "var a=w.clientWidth-32;"
        "var s=Math.max(m,Math.min(1,a/n));"
        "o.style.zoom=s<1?''+s:'';"
        "}"
        "window.addEventListener('resize',fit);"
        "fit();"
        "})();"
    )
    return (
        f"<!DOCTYPE html><html><head><meta charset='UTF-8'>{font}"
        f"<style>{css}</style></head><body>"
        f"<div id='sw'>"
        f"<div id='bk-outer'>"
        f"{header_html}{bracket_html}"
        f"</div>"
        f"</div>"
        f"<script>{js}</script>"
        f"</body></html>"
    )


def render_group_standings() -> None:
    """12-group grid (Group Stage) + knockout bracket tree (Knockout Bracket)."""
    gs_tab, ko_tab = st.tabs(["Group Stage", "Knockout Bracket"])

    with gs_tab:
        try:
            matches = fetch_match_predictions()
            ko_payload = fetch_knockout_bracket()
        except requests.RequestException as exc:
            st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
        else:
            # Build set of confirmed qualifiers from the R32 bracket
            r32 = ko_payload.get("knockout_bracket", {}).get("R32", [])
            qualified_teams: set[str] = set()
            for m in r32:
                if m.get("home"):
                    qualified_teams.add(m["home"])
                if m.get("away"):
                    qualified_teams.add(m["away"])

            # Index matches by group letter
            group_matches: dict[str, list[dict]] = {letter: [] for letter in GROUPS}
            for m in matches:
                g = m.get("group")
                if g and g in group_matches:
                    group_matches[g].append(m)

            st.markdown(
                '<p class="wc-eyebrow">12 groups &middot; 48 teams &middot; '
                'Real results for played matches, model predictions for upcoming fixtures.</p>',
                unsafe_allow_html=True,
            )

            # Legend
            st.markdown(
                f'<div class="wc-legend" style="flex-wrap:wrap;gap:10px 20px;margin-bottom:12px;">'
                f'<span><span style="display:inline-block;width:16px;height:16px;line-height:16px;'
                f'text-align:center;font-size:9px;font-weight:800;border-radius:3px;'
                f'background:{STATUS_OK};color:#0d2a1a;">Q</span>&nbsp;Qualified for R32</span>'
                f'<span><span style="display:inline-block;width:16px;height:16px;line-height:16px;'
                f'text-align:center;font-size:9px;font-weight:800;border-radius:3px;'
                f'background:{GREY};color:#1a1a1a;">?</span>&nbsp;TBD</span>'
                f'<span><span style="display:inline-block;width:16px;height:16px;line-height:16px;'
                f'text-align:center;font-size:9px;font-weight:800;border-radius:3px;'
                f'background:{RED};color:#2a0a06;">E</span>&nbsp;Eliminated</span>'
                f'<span style="color:{TXT3};font-size:12px;">'
                f'W/D/L &middot; GF/GA/GD &middot; Pts'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            _stat_cols = ["W", "D", "L", "GF", "GA", "GD", "Pts"]
            _col_w     = [24,  24,  24,  30,   30,   34,   32  ]

            cards: list[str] = []
            for letter, teams in GROUPS.items():
                gm = group_matches[letter]
                group_complete = len(gm) > 0 and all(m.get("status") == "played" for m in gm)

                # Accumulate real results
                stats: dict[str, dict] = {
                    t: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}
                    for t in teams
                }
                for m in gm:
                    if m.get("status") != "played":
                        continue
                    h, a = m["home"], m["away"]
                    hs, as_ = int(m["home_score"]), int(m["away_score"])
                    for t, gf, ga in ((h, hs, as_), (a, as_, hs)):
                        if t not in stats:
                            continue
                        stats[t]["P"]  += 1
                        stats[t]["GF"] += gf
                        stats[t]["GA"] += ga
                        if gf > ga:
                            stats[t]["W"]   += 1
                            stats[t]["Pts"] += 3
                        elif gf == ga:
                            stats[t]["D"]   += 1
                            stats[t]["Pts"] += 1
                        else:
                            stats[t]["L"] += 1

                ranked = sorted(
                    teams,
                    key=lambda t: (stats[t]["Pts"], stats[t]["GF"] - stats[t]["GA"], stats[t]["GF"]),
                    reverse=True,
                )

                # Column header row
                hdr_cells = "".join(
                    f'<span style="min-width:{w}px;text-align:right;font-size:10px;'
                    f'font-weight:700;letter-spacing:1px;color:{TXT3};">{c}</span>'
                    for c, w in zip(_stat_cols, _col_w)
                )
                hdr = (
                    f'<div style="display:flex;align-items:center;gap:2px;'
                    f'padding:5px 8px 5px 4px;border-bottom:1px solid {CARD_BD};">'
                    f'<span style="min-width:24px;flex-shrink:0;"></span>'
                    f'<span style="width:28px;flex-shrink:0;"></span>'
                    f'<span style="flex:1;"></span>'
                    f'{hdr_cells}'
                    f'</div>'
                )

                rows: list[str] = []
                for t in ranked:
                    s = stats[t]
                    gd = s["GF"] - s["GA"]
                    gd_str = f'+{gd}' if gd > 0 else str(gd)

                    if t in qualified_teams:
                        badge_bg, badge_c, badge_txt = STATUS_OK, "#0d2a1a", "Q"
                    elif group_complete and t not in qualified_teams:
                        badge_bg, badge_c, badge_txt = RED, "#2a0a06", "E"
                    else:
                        badge_bg, badge_c, badge_txt = GREY, "#1a1a1a", "?"

                    badge = (
                        f'<span style="display:inline-block;width:24px;height:24px;'
                        f'line-height:24px;text-align:center;font-size:13px;font-weight:800;'
                        f'border-radius:4px;background:{badge_bg};color:{badge_c};'
                        f'flex-shrink:0;">{badge_txt}</span>'
                    )

                    vals = [s["W"], s["D"], s["L"], s["GF"], s["GA"], gd, s["Pts"]]
                    stat_cells = "".join(
                        f'<span style="min-width:{w}px;text-align:right;'
                        f'font-size:13px;font-family:\'Barlow Condensed\',sans-serif;font-weight:700;'
                        f'color:{TXT if (col == "Pts" or v != 0) else TXT3};">'
                        f'{gd_str if col == "GD" else v}</span>'
                        for (col, w), v in zip(zip(_stat_cols, _col_w), vals)
                    )

                    rows.append(
                        f'<div style="display:flex;align-items:center;gap:2px;'
                        f'padding:8px 8px 8px 4px;" title="{html.escape(t)}">'
                        f'{badge}'
                        f'{flag_img(t, 24)}'
                        f'<span style="flex:1;"></span>'
                        f'{stat_cells}'
                        f'</div>'
                    )

                cards.append(
                    f'<div class="wc-group">'
                    f'  <div class="wc-group-hd">'
                    f'    <span class="wc-group-ltr">GROUP {letter}</span>'
                    f'  </div>'
                    f'  {hdr}'
                    f'  {"".join(rows)}'
                    f'</div>'
                )

            st.markdown(f'<div class="wc-groups">{"".join(cards)}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="wc-foot">Standings from real played results &middot; '
                'Q/E badges update when bracket data is confirmed &middot; '
                'Tiebreakers: GD then GF.</div>',
                unsafe_allow_html=True,
            )

    with ko_tab:
        try:
            ko_payload = fetch_knockout_bracket()
            bracket    = ko_payload["knockout_bracket"]
        except requests.RequestException as exc:
            st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
        else:
            st.markdown(
                '<p class="wc-eyebrow">Left-to-right bracket &middot; '
                '<span style="color:#9FD45B;font-weight:600;">Green border</span> = locked result &middot; '
                'Percentages = model predictions for upcoming matches.</p>',
                unsafe_allow_html=True,
            )
            components.html(_build_bracket_tree_html(bracket), height=1560, scrolling=False)


# ---------------------------------------------------------------------------
# Page 4 — Odds Tracker
# ---------------------------------------------------------------------------

def team_colors(team: str) -> tuple[str, str]:
    """Return a team's (primary, secondary) flag colors, falling back to the brand palette."""
    return TEAM_COLORS.get(team, (GREEN, GOLD))


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _color_distance(a: str, b: str) -> float:
    """Euclidean distance between two hex colors in RGB space (0–441)."""
    r1, g1, b1 = _hex_to_rgb(a)
    r2, g2, b2 = _hex_to_rgb(b)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


def resolve_track_styles(teams: list[str], odds: dict[str, float]) -> dict[str, dict]:
    """
    Assign each tracked team a line style for the odds chart.

    Teams are processed by title odds (highest first). Each gets a solid primary-color
    line by default; if its primary is too close to a color already on the chart, it
    falls back to the two-color treatment (solid primary base + dashed secondary) so it
    stays visually distinct from its look-alike.

    Returns a per-team dict: {"mode": "solid"|"split", "primary": hex, "secondary": hex}.
    """
    _THRESHOLD = 80
    sorted_teams = sorted(teams, key=lambda t: odds.get(t, 0), reverse=True)
    occupied: list[str] = []
    styles: dict[str, dict] = {}
    for team in sorted_teams:
        primary, secondary = team_colors(team)
        if any(_color_distance(primary, c) < _THRESHOLD for c in occupied):
            styles[team] = {"mode": "split", "primary": primary, "secondary": secondary}
            occupied.extend((primary, secondary))
        else:
            styles[team] = {"mode": "solid", "primary": primary, "secondary": secondary}
            occupied.append(primary)
    return styles


def _compute_movers(history: list[dict]) -> tuple[tuple, tuple] | None:
    first, last = history[0]["tournament_odds"], history[-1]["tournament_odds"]
    deltas = {t: (last[t] - first.get(t, 0.0)) * 100 for t in last}
    if not deltas:
        return None
    return max(deltas.items(), key=lambda kv: kv[1]), min(deltas.items(), key=lambda kv: kv[1])


def render_odds_tracker() -> None:
    """Biggest-mover cards + multi-line title-probability Altair chart."""
    try:
        history = fetch_odds_history()
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
        return

    st.markdown(
        '<p class="wc-eyebrow">Title probability per team across nightly model re-runs.</p>',
        unsafe_allow_html=True,
    )

    if len(history) < 2:
        st.info("Not enough snapshots yet — check back after the next retrain.")
        return

    movers = _compute_movers(history)
    if movers:
        (rt, rd), (ft, fd) = movers
        st.markdown(
            f"""
            <div class="wc-movers">
              <div class="wc-mover">
                <div class="lbl">{trend_up_svg()} Biggest riser &middot; since first snapshot</div>
                <div class="bd">{flag_img(rt)} {html.escape(rt)}
                  <span style="margin-left:auto;font-family:'Barlow Condensed',sans-serif;
                    font-size:22px;font-weight:800;color:{GREEN_TXT};">+{rd:.1f} pp</span>
                </div>
              </div>
              <div class="wc-mover">
                <div class="lbl">{trend_down_svg()} Biggest faller &middot; since first snapshot</div>
                <div class="bd">{flag_img(ft)} {html.escape(ft)}
                  <span style="margin-left:auto;font-family:'Barlow Condensed',sans-serif;
                    font-size:22px;font-weight:800;color:{RED};">{fd:.1f} pp</span>
                </div>
              </div>
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
    tournament_end = pd.Timestamp("2026-07-21", tz="UTC")
    df = df[df["generated_at"] < tournament_end]

    latest        = history[-1]["tournament_odds"]
    default_teams = sorted(latest, key=latest.get, reverse=True)[:6]
    teams         = st.multiselect("Teams to track", sorted(df["team"].unique()), default=default_teams)
    if not teams:
        st.info("Select at least one team to plot.")
        return

    # Higher-ranked teams get a solid primary line; teams whose primary clashes with a
    # color already on the chart fall back to a two-tone (primary + dashed secondary)
    # line so they stay distinct. The chips double as the legend, with a swatch that
    # matches each team's rendering (single color vs. split).
    styles      = resolve_track_styles(teams, latest)
    split_teams = [t for t, s in styles.items() if s["mode"] == "split"]

    def _swatch(team: str) -> str:
        s = styles[team]
        bg = (f'linear-gradient(135deg,{s["primary"]} 0 50%,{s["secondary"]} 50% 100%)'
              if s["mode"] == "split" else s["primary"])
        return (
            f'<span style="display:inline-block;width:14px;height:14px;border-radius:3px;'
            f'background:{bg};border:0.5px solid rgba(255,255,255,0.22);'
            f'vertical-align:middle;flex:none;"></span>'
        )

    chips = "".join(
        f'<span class="wc-chip">{_swatch(t)}'
        f' {flag_img(t, 18)} {html.escape(t)}'
        f' <span style="color:{TXT3};">{latest.get(t, 0) * 100:.1f}%</span></span>'
        for t in sorted(teams, key=lambda t: latest.get(t, 0), reverse=True)
    )
    st.markdown(f'<div class="wc-chips">{chips}</div>', unsafe_allow_html=True)

    domain        = list(teams)
    primary_range = [styles[t]["primary"] for t in teams]

    axis_kw = dict(labelColor=TXT2, titleColor=TXT2, gridColor=CARD_BD,
                   domainColor=CARD_BD, tickColor=CARD_BD)
    base = alt.Chart(df[df["team"].isin(teams)]).encode(
        x=alt.X("generated_at:T", title="Snapshot", axis=alt.Axis(**axis_kw)),
        y=alt.Y("p:Q", title="Title probability", axis=alt.Axis(format="%", **axis_kw)),
        detail="team:N",
    )
    # Solid base line in each flag's primary color — every team shows this.
    primary_line = base.mark_line(strokeWidth=3.5).encode(
        color=alt.Color("team:N",
                        scale=alt.Scale(domain=domain, range=primary_range),
                        legend=None),
    )
    layers = [primary_line]

    # Dashed secondary overlay, drawn only for teams flagged as clashing. The gaps
    # reveal the primary beneath, giving those teams a bicolor line.
    if split_teams:
        secondary_range = [styles[t]["secondary"] for t in split_teams]
        secondary_line = (
            base.transform_filter(alt.FieldOneOfPredicate("team", split_teams))
            .mark_line(strokeWidth=4, strokeDash=[10, 10])
            .encode(color=alt.Color("team:N",
                                    scale=alt.Scale(domain=split_teams, range=secondary_range),
                                    legend=None))
        )
        layers.append(secondary_line)

    points = base.mark_point(size=55, filled=True).encode(
        color=alt.Color("team:N",
                        scale=alt.Scale(domain=domain, range=primary_range),
                        legend=None),
        tooltip=[
            "team",
            alt.Tooltip("generated_at:T", title="Snapshot"),
            alt.Tooltip("p:Q", title="Title probability", format=".1%"),
        ],
    )
    layers.append(points)

    chart = (
        alt.layer(*layers)
        .resolve_scale(color="independent")
        .properties(height=480)
        .configure(background="transparent")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True, theme=None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    inject_theme()
    render_header()
    render_banner()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Tournament Odds", "Match Predictor", "Standings & Bracket", "Odds Tracker"]
    )
    with tab1:
        render_tournament_odds()
    with tab2:
        render_match_predictor()
    with tab3:
        render_group_standings()
    with tab4:
        render_odds_tracker()


if __name__ == "__main__":
    main()
