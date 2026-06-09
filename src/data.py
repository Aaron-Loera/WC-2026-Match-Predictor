from __future__ import annotations
import os
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup 
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

RANKINGS_STALENESS_DAYS = 30
MATCHES_STALENESS_DAYS = 1
WC_STALENESS_HOURS = 6

_ELO_RANKINGS_URL = "https://www.eloratings.net/World.tsv"
_ELO_NAMES_URL = "https://www.eloratings.net/en.teams.tsv"

# Confederation lookup keyed by eloratings.net country code
_CONFEDERATION: dict[str, str] = {
    # UEFA
    "AL": "UEFA", "AD": "UEFA", "AM": "UEFA", "AT": "UEFA", "AZ": "UEFA",
    "BY": "UEFA", "BE": "UEFA", "BA": "UEFA", "BG": "UEFA", "HR": "UEFA",
    "CY": "UEFA", "CS": "UEFA", "CZ": "UEFA", "DK": "UEFA", "EN": "UEFA",
    "EE": "UEFA", "FI": "UEFA", "FR": "UEFA", "GE": "UEFA", "DE": "UEFA",
    "GR": "UEFA", "HU": "UEFA", "IS": "UEFA", "IE": "UEFA", "IL": "UEFA",
    "IT": "UEFA", "KZ": "UEFA", "LV": "UEFA", "LI": "UEFA", "LT": "UEFA",
    "LU": "UEFA", "MT": "UEFA", "MD": "UEFA", "ME": "UEFA", "NL": "UEFA",
    "MK": "UEFA", "NO": "UEFA", "PL": "UEFA", "PT": "UEFA", "RO": "UEFA",
    "RU": "UEFA", "SM": "UEFA", "SC": "UEFA", "RS": "UEFA", "SK": "UEFA",
    "SI": "UEFA", "ES": "UEFA", "SE": "UEFA", "CH": "UEFA", "TR": "UEFA",
    "UA": "UEFA", "GB": "UEFA", "WA": "UEFA",
    # CONMEBOL
    "AR": "CONMEBOL", "BO": "CONMEBOL", "BR": "CONMEBOL", "CL": "CONMEBOL",
    "CO": "CONMEBOL", "EC": "CONMEBOL", "PY": "CONMEBOL", "PE": "CONMEBOL",
    "UY": "CONMEBOL", "VE": "CONMEBOL",
    # CONCACAF
    "AG": "CONCACAF", "BS": "CONCACAF", "BB": "CONCACAF", "BZ": "CONCACAF",
    "CA": "CONCACAF", "CR": "CONCACAF", "CU": "CONCACAF", "DM": "CONCACAF",
    "DO": "CONCACAF", "SV": "CONCACAF", "GD": "CONCACAF", "GT": "CONCACAF",
    "HT": "CONCACAF", "HN": "CONCACAF", "JM": "CONCACAF", "MX": "CONCACAF",
    "NI": "CONCACAF", "PA": "CONCACAF", "KN": "CONCACAF", "LC": "CONCACAF",
    "VC": "CONCACAF", "TT": "CONCACAF", "US": "CONCACAF",
    # CAF
    "DZ": "CAF", "AO": "CAF", "BJ": "CAF", "BW": "CAF", "BF": "CAF",
    "BI": "CAF", "CM": "CAF", "CV": "CAF", "CF": "CAF", "TD": "CAF",
    "CG": "CAF", "CD": "CAF", "CI": "CAF", "DJ": "CAF", "EG": "CAF",
    "GQ": "CAF", "ER": "CAF", "ET": "CAF", "GA": "CAF", "GM": "CAF",
    "GH": "CAF", "GN": "CAF", "GW": "CAF", "KE": "CAF", "LS": "CAF",
    "LR": "CAF", "LY": "CAF", "MG": "CAF", "MW": "CAF", "ML": "CAF",
    "MR": "CAF", "MU": "CAF", "MA": "CAF", "MZ": "CAF", "NA": "CAF",
    "NE": "CAF", "NG": "CAF", "RW": "CAF", "ST": "CAF", "SN": "CAF",
    "SL": "CAF", "SO": "CAF", "ZA": "CAF", "SS": "CAF", "SD": "CAF",
    "SZ": "CAF", "TZ": "CAF", "TG": "CAF", "TN": "CAF", "UG": "CAF",
    "ZM": "CAF", "ZW": "CAF",
    # AFC
    "AF": "AFC", "AU": "AFC", "BH": "AFC", "BD": "AFC", "BT": "AFC",
    "BN": "AFC", "KH": "AFC", "CN": "AFC", "GU": "AFC", "HK": "AFC",
    "IN": "AFC", "ID": "AFC", "IR": "AFC", "IQ": "AFC", "JP": "AFC",
    "JO": "AFC", "KW": "AFC", "KG": "AFC", "LA": "AFC", "LB": "AFC",
    "MO": "AFC", "MY": "AFC", "MV": "AFC", "MN": "AFC", "MM": "AFC",
    "NP": "AFC", "KP": "AFC", "OM": "AFC", "PK": "AFC", "PS": "AFC",
    "PH": "AFC", "QA": "AFC", "SA": "AFC", "SG": "AFC", "KR": "AFC",
    "LK": "AFC", "SY": "AFC", "TW": "AFC", "TJ": "AFC", "TH": "AFC",
    "TL": "AFC", "TM": "AFC", "AE": "AFC", "UZ": "AFC", "VN": "AFC",
    "YE": "AFC",
    # OFC
    "CK": "OFC", "FJ": "OFC", "KI": "OFC", "MH": "OFC", "FM": "OFC",
    "NR": "OFC", "NZ": "OFC", "PW": "OFC", "PG": "OFC", "WS": "OFC",
    "SB": "OFC", "TO": "OFC", "TV": "OFC", "VU": "OFC",
}
_MATCHES_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
_WC2026_URL = "https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026/"

_WC_RESULTS_SCHEMA = {
    "date": "datetime64[ns]",
    "home_team": "object",
    "away_team": "object",
    "home_score": "Int64",
    "away_score": "Int64",
    "stage": "object",
    "match_id": "object",
}


def _is_stale(path: Path, max_age_seconds: float) -> bool:
    """
    Determines if a file is stale by checking if it's missing or if its last modified
    time is greater than `max_age_seconds`.
    
    Args:
        path: Path to the cached file to check.
        max_age_seconds: Max acceptable age of the file in seconds.
        
    Returns:
        bool: True if the file does not exist or if its older than `max_age_seconds`.
    """
    if not path.exists():
        return True
    return time.time() - path.stat().st_mtime > max_age_seconds


def fetch_fifa_rankings() -> pd.DataFrame:
    """
    Fetch Elo-based football team rankings from "eloratings.net".
    
    Returns a cached copy if it's not stale. Otherwise, fresh data is fetched from
    "eloratings.net" TSV feeds.
    
    Args:
        None:
    
    Returns:
        pd.DataFrame: A DataFrame with columns [rank, team, total_points, confederation].
    
    """
    cache = RAW_DIR / "fifa_rankings.csv"
    max_age = RANKINGS_STALENESS_DAYS * 86400

    # Return CSV file if rankings are not stable
    if not _is_stale(cache, max_age):
        return pd.read_csv(cache)  

    ua_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/tab-separated-values,text/plain,*/*"
    }

    code_to_name: dict[str, str] = {}
    
    # Attempt to fetch {code: name} dictionary from eloratings.net
    try:
        names_resp = requests.get(_ELO_NAMES_URL, headers=ua_headers, timeout=30)
        names_resp.raise_for_status()
        for line in names_resp.text.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and not parts[0].endswith("_loc"):
                code_to_name[parts[0]] = parts[1]
                
    except requests.RequestException:
        print(f"Warning: could not fetch Elo team names from {_ELO_NAMES_URL}")
        code_to_name = {}

    # Fetch (rank, code, rating) rows from rankings
    rankings_resp = requests.get(_ELO_RANKINGS_URL, headers=ua_headers, timeout=30)
    rankings_resp.raise_for_status()

    # Assign team information
    rows = []
    for line in rankings_resp.text.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        code = parts[2]
        name = code_to_name.get(code, code)
        rows.append({
            "rank": parts[0],
            "team": name,
            "total_points": parts[3],
            "confederation": _CONFEDERATION.get(code, ""),
        })

    # Empty row guard
    if not rows:
        preview = rankings_resp.text[:400].replace("\n", "\\n")
        raise RuntimeError(
            f"fetch_fifa_rankings: 0 rows parsed from {_ELO_RANKINGS_URL} — "
            f"site may be blocking runner IP or TSV format changed. "
            f"Response preview: {preview!r}"
        )

    # Create and standardize DataFrame
    df = pd.DataFrame(rows)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")

    # Create directory, save rankings, return frame
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def fetch_historical_matches(n_years: int) -> pd.DataFrame:
    """
    Fetch international football match results for the last `n_years` years.
    
    Return a cached copy if it's not stale. Otherwise, download the full results CSV
    from the "martj42/international_results" repository.
    
    Args:
        n_years: Number of years of history to return, starting from today.
        
    Returns:
        pd.DataFrame: A DataFrame with columns [date, home_team, away_team, home_score, away_score, tournament, neutral].
    """
    cache = RAW_DIR / "historical_matches.parquet"
    max_age = MATCHES_STALENESS_DAYS * 86400

    # Return frame if cache isn't stale
    if not _is_stale(cache, max_age):
        df = pd.read_parquet(cache, engine="auto")
        cutoff = datetime.now() - timedelta(days=365 * n_years)
        return df[df["date"] >= cutoff].reset_index(drop=True)

    # Fetch historical CSV metadata from matches
    response = requests.get(_MATCHES_URL, timeout=60)
    response.raise_for_status()

    # Read CSV file into frame
    from io import StringIO
    raw = pd.read_csv(StringIO(response.text))

    # Standardize frame
    df = raw.rename(columns={
        "home_score": "home_score",
        "away_score": "away_score"
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "home_score" in df.columns:
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
    if "away_score" in df.columns:
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")
    df["neutral"] = df["neutral"].astype(bool) if "neutral" in df.columns else False

    keep = [c for c in ("date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral") if c in df.columns]
    df = df[keep]

    # Create directory and save parquet
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, engine="auto", compression="snappy", index=False)

    # Return frame with applied cutoff
    cutoff = datetime.now() - timedelta(days=365 * n_years)
    return df[df["date"] >= cutoff].reset_index(drop=True)


def fetch_wc_results() -> pd.DataFrame:
    """
    Scrape completed FIFA World Cup 2026 match results from the FIFA website.
    
    Returns a cached copy if it's not stale. If the HTTP request fails or the tournament has
    not yet started, an empty DataFrame is returned.
    
    Args:
        None:
        
    Returns:
        pd.DataFrame: A DataFrame with columns [data, home_team, away_team, home_score, away_score, stage, match_id].
    """
    cache = RAW_DIR / "wc2026_results.parquet"
    max_age = WC_STALENESS_HOURS * 3600

    # Returns parquet file if cache isn't stale
    if not _is_stale(cache, max_age):
        return pd.read_parquet(cache, engine="auto")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        # Attempt to fetch match results
        response = requests.get(_WC2026_URL, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")     
    except requests.RequestException:
        # Return empty frame if scrape fails (tournament may not be live yet)
        return _empty_wc_df()

    rows: list[dict] = []
    
    # 3 CSS-selector strategy targeting completed match card elements
    match_cards = soup.select("[class*='match'][class*='completed']") or \
                  soup.select("article[data-matchstatus='FINISHED']") or \
                  soup.select("[data-match-status='FINISHED']")

    for card in match_cards:
        try:
            # Attempt to parse each card
            home = card.select_one("[class*='homeTeam'] [class*='name']")
            away = card.select_one("[class*='awayTeam'] [class*='name']")
            home_score = card.select_one("[class*='homeScore']") or card.select_one("[class*='score']:first-child")
            away_score = card.select_one("[class*='awayScore']") or card.select_one("[class*='score']:last-child")
            date_el = card.select_one("[class*='date']") or card.select_one("time")
            stage_el = card.select_one("[class*='stage']") or card.select_one("[class*='round']")
            match_id = card.get("data-match-id") or card.get("id") or ""

            if not (home and away and home_score and away_score):
                continue

            rows.append({
                "date": pd.to_datetime(date_el.get_text(strip=True) if date_el else "", errors="coerce"),
                "home_team": home.get_text(strip=True),
                "away_team": away.get_text(strip=True),
                "home_score": int(home_score.get_text(strip=True)),
                "away_score": int(away_score.get_text(strip=True)),
                "stage": stage_el.get_text(strip=True) if stage_el else "",
                "match_id": str(match_id),
            })   
        except (ValueError, AttributeError):
            continue

    if not rows:
        df = _empty_wc_df()
    else:
        # Create and standardize frame
        df = pd.DataFrame(rows)
        df["home_score"] = df["home_score"].astype("Int64")
        df["away_score"] = df["away_score"].astype("Int64")

    # Create directory, save parquet, and return frame
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, engine="auto", compression="snappy", index=False)
    return df


def _empty_wc_df() -> pd.DataFrame:
    """Returns an empty DataFrame with pre-established dtypes."""
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in _WC_RESULTS_SCHEMA.items()})


def save_processed(df: pd.DataFrame, name: str) -> None:
    """
    Save a processed DataFrame as a parquet file under the provided filename.
    
    Args:
        pd.DataFrame: The processed DataFrame to save.
        name: The filename to be saved under.
    
    Returns:
        None:   
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / f"{name}.parquet", engine="pyarrow", index=False)


def load_processed(name: str) -> pd.DataFrame:
    """
    Load a parquet file with the provided filename as a DataFrame.
    
    Args:
        name: The name of the parquet file.
        
    Returns:
        pd.DataFrame: A processed DataFrame.
    """
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed file '{name}' — run features.py first.")
    return pd.read_parquet(path, engine="auto")


if __name__ == "__main__":
    import argparse

    # Parse "--update" flag from CLI
    parser = argparse.ArgumentParser(description="Fetch and cache raw WC2026 data.")
    parser.add_argument("--update", action="store_true", help="Force re-fetch all sources")
    args = parser.parse_args()

    # Delete caches to force re-fetch on next call
    if args.update:
        for f in [
            RAW_DIR / "fifa_rankings.csv",
            RAW_DIR / "historical_matches.parquet",
            RAW_DIR / "wc2026_results.parquet",
        ]:
            if f.exists():
                f.unlink()

    rankings = fetch_fifa_rankings()
    print(f"FIFA rankings: {len(rankings)} teams")

    matches = fetch_historical_matches(n_years=25)
    print(f"Historical matches: {len(matches)} rows")

    wc = fetch_wc_results()
    print(f"WC2026 results: {len(wc)} matches so far")
