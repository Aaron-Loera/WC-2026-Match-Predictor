"""
Streamlit dashboard for the WC 2026 Match Predictor — redesigned.

Pages (st.tabs):
  1. Tournament Odds  — Countdown banner · KPI strip · animated title-race bars.
  2. Match Predictor  — Team selectors · split bar · outcome cards.
  3. Group Standings  — 12-group grid ranked by title probability.
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

API_BASE_URL        = os.getenv("API_BASE_URL", "http://localhost:8000")
CACHE_TTL_SECONDS   = int(os.getenv("DASHBOARD_CACHE_TTL", "3600"))
REQUEST_TIMEOUT     = 60

KICKOFF_DATE = dt.date(2026, 6, 11)
KICKOFF_DT   = dt.datetime(2026, 6, 11, 18, 0, 0, tzinfo=dt.timezone.utc)

# Palette — single source of truth for inline HTML/f-strings
BG       = "#1B1D1F"; CARD     = "#26292C"; CARD_BD  = "#34383C"; TRACK = "#303438"
TXT      = "#E8EAEC"; TXT2     = "#9AA0A6"; TXT3     = "#7A8087"
GREEN    = "#7FB83E"; GREEN_TXT = "#9FD45B"; GREEN_TINT = "#2A3A18"
GOLD     = "#F2A93B"; GOLD_TXT  = "#F2B85C"; GREY     = "#80868C"; RED = "#E5705B"
STATUS_OK = "#3BD68B"; STATUS_WARN = "#F2A93B"; STATUS_ERR = "#E5705B"

_ICON_PATH = Path(__file__).resolve().parent / "assets" / "ball_icon.svg"
PAGE_ICON  = str(_ICON_PATH) if _ICON_PATH.exists() else "⚽"

st.set_page_config(
    page_title="WC 2026 Match Predictor",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Team reference data
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
# HTML helpers
# ---------------------------------------------------------------------------

def ball_svg(size: int = 22) -> str:
    """Inline green soccer-ball brand mark as SVG."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 582 582" '
        'xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;flex:none;">'
        '<defs><linearGradient id="bG" gradientUnits="userSpaceOnUse" x1="291" y1="32" x2="291" y2="550">'
        '<stop offset="0" stop-color="#9EDB35"/><stop offset="1" stop-color="#5DA10C"/>'
        '</linearGradient></defs>'
        '<circle cx="291" cy="291" r="259" fill="none" stroke="url(#bG)" stroke-width="63"/>'
        '<path d="M291,148.5 L426.5,247 L374.8,406.3 L207.2,406.3 L155.5,247 Z" '
        'fill="none" stroke="url(#bG)" stroke-width="63" stroke-linejoin="round"/>'
        '<g stroke="url(#bG)" stroke-width="63">'
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
    """Return an <img> tag for a team's flag, or a blank placeholder if unknown."""
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
    """Return a team's confederation acronym, or '' if unknown."""
    return TEAM_CONFED.get(team, "")


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _get(endpoint: str) -> dict | list:
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


@st.cache_data(ttl=60)
def fetch_health() -> dict:
    """Short-TTL wrapper around GET /health so the status dot stays fresh."""
    return _get("/health")


def _format_generated(raw: str | None) -> str:
    """Render an ISO timestamp as 'Jun 7, 2026 20:33 UTC'."""
    if not raw:
        return "unknown"
    try:
        ts = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return ts.strftime("%b %-d, %Y %H:%M UTC")
    except (ValueError, TypeError):
        return raw


# ---------------------------------------------------------------------------
# Shared chrome
# ---------------------------------------------------------------------------

def render_header() -> None:
    """Brand row + API health dot (no sidebar)."""
    dot, label = STATUS_WARN, "API status unknown"
    try:
        health = fetch_health()
        if (health.get("status") == "ok"
                and health.get("model_loaded")
                and health.get("predictions_loaded")):
            dot, label = STATUS_OK, "API live"
        else:
            dot, label = STATUS_WARN, "API degraded"
        label = f"{label} · predictions {_format_generated(health.get('generated_at'))}"
    except requests.RequestException:
        dot, label = STATUS_ERR, "API unreachable"

    st.markdown(
        f'<div class="wc-header">'
        f'  <div class="wc-brand">{ball_svg(22)} WC 2026 Predictor</div>'
        f'  <div class="wc-api-status">'
        f'    <span class="wc-dot" style="background:{dot};"></span>'
        f'    {html.escape(label)}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _banner_component_html(ticker: str) -> str:
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
        "el.innerHTML='<div class=\"wc-countdown\" style=\"font-family:Barlow Condensed,sans-serif;"
        "font-size:40px;font-weight:800;color:#9FD45B;letter-spacing:2px;\">UNDERWAY</div>';"
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
        "</script></body></html>"
    )


def render_banner() -> None:
    """Countdown banner with live JS tick, pitch-stripe texture, and title-odds ticker."""
    try:
        payload = fetch_tournament_odds()
        odds: dict[str, float] = payload["tournament_odds"]
    except requests.RequestException:
        odds = {}

    top8 = sorted(odds.items(), key=lambda kv: kv[1], reverse=True)[:8]
    ticker = "".join(
        f'<span class="wc-ticker-item">{flag_img(t, 16)}'
        f' <strong>{html.escape(t)}</strong>'
        f' <span class="wc-tick-val">{p * 100:.1f}%</span></span>'
        for t, p in top8
    ) * 2

    components.html(_banner_component_html(ticker), height=120, scrolling=False)


# ---------------------------------------------------------------------------
# Page 1 — Tournament Odds
# ---------------------------------------------------------------------------

def render_tournament_odds() -> None:
    """KPI strip + pitch-divider + animated title-race bars (top 16)."""
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
    n_sims   = payload.get("n_simulations", 0)
    gen_full = _format_generated(payload.get("generated_at"))

    st.markdown(
        f'<p class="wc-eyebrow">Each team\'s probability of lifting the trophy, '
        f'across {n_sims:,} simulated tournaments.</p>',
        unsafe_allow_html=True,
    )

    fav      = df.iloc[0]
    days_left = (KICKOFF_DATE - dt.date.today()).days
    ko_val   = ("Underway" if days_left < 0
                else ("Today" if days_left == 0 else f"{days_left} days"))

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
            <div class="sub">Monte Carlo runs</div>
          </div>
          <div class="wc-kpi">
            <div class="lbl">Field</div>
            <div class="val">{len(df)}</div>
            <div class="sub">teams qualified</div>
          </div>
          <div class="wc-kpi">
            <div class="lbl">Kick-off</div>
            <div class="val" style="font-size:36px;">{html.escape(ko_val)}</div>
            <div class="sub">Jun 11, opener</div>
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

    shown = df.head(16)
    max_p = float(df["p"].max()) or 1.0
    rows: list[str] = []
    for i, r in shown.iterrows():
        if i == 8:
            rows.append(
                '<div class="wc-divider">'
                '<span class="ln"></span><span>Chasing pack</span><span class="ln"></span>'
                '</div>'
            )
        gold  = i < 8
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
    st.markdown(
        f'<div class="wc-foot">Showing top {len(shown)} of {len(df)} teams &middot; '
        f'generated {gen_full} &middot; model retrains nightly.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 2 — Match Predictor
# ---------------------------------------------------------------------------

def render_match_predictor() -> None:
    """Team selectors, split probability bar, outcome cards, most-likely callout."""
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

    col1, col2 = st.columns(2)
    home = col1.selectbox("Home team", sorted(df["home"].unique()))
    away = col2.selectbox(
        "Away team", sorted(df.loc[df["home"] == home, "away"].unique())
    )

    match = df[(df["home"] == home) & (df["away"] == away)].iloc[0]
    ph, pd_, pa = float(match["p_home_win"]), float(match["p_draw"]), float(match["p_away_win"])

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


# ---------------------------------------------------------------------------
# Page 3 — Group Standings
# ---------------------------------------------------------------------------

def render_group_standings() -> None:
    """12-group grid with teams ranked by title-win probability."""
    try:
        payload = fetch_tournament_odds()
        odds: dict[str, float] = payload["tournament_odds"]
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the prediction API at {API_BASE_URL}: {exc}")
        return

    st.markdown(
        '<p class="wc-eyebrow">12 groups &middot; 48 teams &middot; '
        'ranked by title probability. <svg width="8" height="8" viewBox="0 0 8 8" style="vertical-align:middle;">'
        '<circle cx="4" cy="4" r="3" fill="#F2A93B"/></svg> = predicted to advance.</p>',
        unsafe_allow_html=True,
    )

    cards: list[str] = []
    for letter, teams in GROUPS.items():
        sorted_teams = sorted(teams, key=lambda t: odds.get(t, 0), reverse=True)
        rows: list[str] = []
        for i, t in enumerate(sorted_teams):
            p     = odds.get(t, 0)
            top2  = i < 2
            cls   = "wc-gteam top2" if top2 else "wc-gteam"
            color = "#F2A93B" if top2 else "#3A3E42"
            pstr  = f"{p * 100:.1f}%" if p > 0 else "—"
            rows.append(
                f'<div class="{cls}">'
                f'  <svg width="8" height="8" viewBox="0 0 8 8" style="flex-shrink:0;vertical-align:middle;">'
                f'<circle cx="4" cy="4" r="3" fill="{color}"/></svg>'
                f'  {flag_img(t, 18)}'
                f'  <span class="nm">{html.escape(t)}</span>'
                f'  <span class="pct">{pstr}</span>'
                f'</div>'
            )
        cards.append(
            f'<div class="wc-group">'
            f'  <div class="wc-group-hd">'
            f'    <span class="wc-group-ltr">GROUP {letter}</span>'
            f'  </div>'
            f'  {"".join(rows)}'
            f'</div>'
        )

    st.markdown(f'<div class="wc-groups">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="wc-foot">Rankings use title win probability as a proxy for '
        'group-stage quality. Official standings begin Jun 11, 2026.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 4 — Odds Tracker
# ---------------------------------------------------------------------------

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

    latest        = history[-1]["tournament_odds"]
    default_teams = sorted(latest, key=latest.get, reverse=True)[:6]
    teams         = st.multiselect("Teams to track", sorted(df["team"].unique()), default=default_teams)
    if not teams:
        st.info("Select at least one team to plot.")
        return

    chips = "".join(
        f'<span class="wc-chip">{flag_img(t, 18)} {html.escape(t)}'
        f' <span style="color:{TXT3};">{latest.get(t, 0) * 100:.1f}%</span></span>'
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

def main() -> None:
    inject_theme()
    render_header()
    render_banner()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Tournament Odds", "Match Predictor", "Group Standings", "Odds Tracker"]
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
