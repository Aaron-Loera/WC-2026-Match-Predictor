from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
import src.data as data

PROCESSED_DIR = Path("data/processed")
_CONF_LABELS = ["UEFA", "CONMEBOL", "CAF", "CONCACAF", "AFC", "OFC"]

FEATURE_COLS: list[str] = [
    "fifa_rank_diff",
    "form_last5_A",
    "form_last5_B",
    "goals_scored_avg_A",
    "goals_conceded_avg_A",
    "goals_scored_avg_B",
    "goals_conceded_avg_B",
    "wc_experience_A",
    "wc_experience_B",
    "h2h_winrate",
    "rest_days_A",
    "rest_days_B",
    "is_knockout",
    *[f"conf_{c}_A" for c in _CONF_LABELS],
    *[f"conf_{c}_B" for c in _CONF_LABELS],
]

_KNOCKOUT_STAGE_TERMS = ["Round of 16", "Quarter", "Semi", "Final", "3rd"]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_team_history(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenates match rows into long-format team history (two rows per match).
    
    Transforms a DataFrame where each row represents a single match with home and away teams into a
    long-format history where each row represents one team's perspective of one match.
    
    Args:
        matches: DataFrame with columns [date, tournament, home_team, away_team, home_score, away_score]
        
    Returns:
        pd.DataFrame: Long-format history DataFrame with columns [date, tournament, team, opponent, gf, ga, won, drawn]
    """
    # Drop unplayed matches
    completed = matches.dropna(subset=["home_score", "away_score"])

    keep = ["date", "tournament"]

    # Create a "home team" view
    home = completed.assign(
        team=completed["home_team"],
        opponent=completed["away_team"],
        gf=pd.to_numeric(completed["home_score"], errors="coerce"), # goals for
        ga=pd.to_numeric(completed["away_score"], errors="coerce"), # goals against
    )[keep + ["team", "opponent", "gf", "ga"]]

    # Create an "away team" view
    away = completed.assign(
        team=completed["away_team"],
        opponent=completed["home_team"],
        gf=pd.to_numeric(completed["away_score"], errors="coerce"), # goals for
        ga=pd.to_numeric(completed["home_score"], errors="coerce"), # goals against
    )[keep + ["team", "opponent", "gf", "ga"]]

    # Concatenate both views
    history = (
        pd.concat([home, away], ignore_index=True)
        .sort_values(["team", "date"])
        .reset_index(drop=True)
    )
    
    # Derive outcome columns
    history["won"] = history["gf"] > history["ga"]
    history["drawn"] = history["gf"] == history["ga"]
    return history


def _add_rolling_stats(history: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling performance statistics to team history rows.
    
    Computes per-team rolling windows for form (5-match win rate), goals scored/conceded
    (10-match averages), and rest days between matches.
    
    Args:
        history: Output DataFrame from `_build_team_history()`.
        
    Returns:
        pd.DataFrame: Input history with four new columns: [form_last5, goals_scored_avg, goals_conceded_avg, rest_days].
    """
    history = history.sort_values(["team", "date"]).copy()
    
    # Group by team
    grp = history.groupby("team", sort=False)
    
    # Rolling win rate over last 5 matches
    history["form_last5"] = grp["won"].transform(
        lambda s: s.astype(float).shift(1).rolling(5, min_periods=1).mean()
    )
    # Average goals scored over last 10 matches
    history["goals_scored_avg"] = grp["gf"].transform(
        lambda s: s.astype(float).shift(1).rolling(10, min_periods=1).mean()
    )
    # Average goals conceded over last 10 matches
    history["goals_conceded_avg"] = grp["ga"].transform(
        lambda s: s.astype(float).shift(1).rolling(10, min_periods=1).mean()
    )
    # Days since last match
    history["rest_days"] = grp["date"].transform(lambda s: s.diff().dt.days)
    return history


def _add_wc_experience(history: pd.DataFrame) -> pd.DataFrame:
    """
    Add cumulative prior World Cup match appearances to team history rows.

    For each team at each match, compute how many FIFA World Cup tournament matches
    (excluding qualifications) they have played before that row.
    
    Args:
        history: Output DataFrame from `_add_rolling_stats()`.
        
    Returns:
        pd.DataFrame: Input history with new column `wc_experience`.
    """
    history = history.copy()
    
    # Identify actual World Cup matches
    is_wc = (
        history["tournament"].str.contains("FIFA World Cup", na=False)
        & ~history["tournament"].str.contains("qualification", case=False, na=False)
    )
    history["_is_wc"] = is_wc.astype(int)
    
    # Add cumulative WC participation count
    history["wc_experience"] = history.groupby("team", sort=False)["_is_wc"].transform(
        lambda s: s.shift(1).cumsum().fillna(0)
    )
    return history.drop(columns=["_is_wc"])


def _compute_h2h(matches: pd.DataFrame) -> pd.Series:
    """
    Compute the home team's head-to-head win rate against the opponent.
    
    For each match, calculate the home team's historical win rate against a specific opponent
    over the prior 10 years. Fills NaN (no prior matches) with 0.5.

    Args:
        matches: DataFrame with columns [date, home_team, away_team, home_score, away_score].
        
    Returns:
        pd.Series: Float Series with home team's h2h win rate for each match.
    """
    df = matches.copy()
    
    # Normalize team pairs in alphabetical form
    df["_pair_a"] = df[["home_team", "away_team"]].min(axis=1)
    df["_pair_b"] = df[["home_team", "away_team"]].max(axis=1)
    df["_pair_key"] = df["_pair_a"] + " vs " + df["_pair_b"]

    # Win indicator from the perspective of "_pair_a" 
    df["_pair_a_won"] = (
        ((df["home_team"] == df["_pair_a"]) & (df["home_score"] > df["away_score"]))
        | ((df["away_team"] == df["_pair_a"]) & (df["away_score"] > df["home_score"]))
    ).astype(float)

    results: list[float] = []
    for idx, row in df.iterrows():
        # Find prior matches with same opponent
        cutoff = row["date"] - pd.Timedelta(days=365 * 10)
        prior = df[
            (df["_pair_key"] == row["_pair_key"])
            & (df["date"] < row["date"])
            & (df["date"] >= cutoff)
        ]
        # Default to 50% rate if empty
        if prior.empty:
            results.append(0.5)
            continue
        # Compute win rate
        pair_a_winrate = prior["_pair_a_won"].mean()
        # Flip if home team is "_pair_b"
        if row["home_team"] == row["_pair_b"]:
            results.append(1.0 - pair_a_winrate)
        else:
            results.append(pair_a_winrate)

    return pd.Series(results, index=matches.index, dtype=float)


def _is_knockout_flag(tournament: pd.Series, stage: pd.Series | None=None) -> pd.Series:
    """
    Flag whether each match is a knockout-stage tournament match.
    
    Returns 1 for knockout rounds (e.g., Round of 16, Quarter-final), and 0 for group-stage
    or non-tournament matches. It no `stage` column is provided, returns all zeros.
    
    Args:
        tournament: Series of tournament names.
        
    Returns:
        pd.Series: Binary integer Series.
    """
    if stage is not None:
        # Build a regex pattern and return binary Series
        pattern = "|".join(_KNOCKOUT_STAGE_TERMS)
        return stage.str.contains(pattern, na=False).astype(int)
    return pd.Series(0, index=tournament.index, dtype=int)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rolling_stats(df: pd.DataFrame, team: str, window: int) -> pd.DataFrame:
    """
    Compute rolling performance metrics for a single team over a specified window.
    
    Given a team-history DataFrame (long-format, one row per team per match), calculate the rolling
    averages for win rate, goals scored, and goals conceded.
    
    Args:
        df: Output of `_build_team_history()`.
        team: Team name to filter for.
        window: Rolling window size.

    Returns:
        pd.DataFrame: DataFrame with columns [form_winrate, goals_scored_avg, goals_conceded_avg].
    """
    # Extract rows for target team
    team_df = df[df["team"] == team].sort_values("date").copy()
    won = team_df["won"].astype(float)
    gf = team_df["gf"].astype(float) # goals for
    ga = team_df["ga"].astype(float) # goals against

    return pd.DataFrame(
        {
            "form_winrate": won.shift(1).rolling(window, min_periods=1).mean().values,
            "goals_scored_avg": gf.shift(1).rolling(window, min_periods=1).mean().values,
            "goals_conceded_avg": ga.shift(1).rolling(window, min_periods=1).mean().values,
        },
        index=team_df["date"].values
    )


def encode_confederation(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode `confederation_A` and `confederation_B` columns.

    Convert the confederation string columns into stable binary features. Guarantees a
    stable, fixed set of output columns.

    Args:
        df: DataFrame with string columns `confederation_A` and `confederation_B`.

    Returns:
        pd.DataFrame: DataFrame with original confederation A/B columns replaced by 6 binary columns per side.
    """
    df = df.copy()
    for side in ("A", "B"):
        col = f"confederation_{side}"
        # One-hot encoding for confederation labels
        for label in _CONF_LABELS:
            df[f"conf_{label}_{side}"] = (df[col] == label).astype(int)
        # Remove original string column
        df = df.drop(columns=[col])
    return df


def build_feature_matrix(matches_df: pd.DataFrame, rankings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw match and rankings DataFrames into an ML-ready feature matrix.

    Orchestrates the full feature-engineering pipeline:
    - Builds per-team rolling history
    - Pivots back to match rows
    - Joins FIFA rankings
    - Computes derived features

    Args:
        matches_df: Output of `data.fetch_historical_matches()`.
        rankings_df: Output of `data.fetch_fifa_rankings()`.

    Returns:
        pd.DataFrame: Feature matrix DataFrame with columns `FEATURE_COLS` + ['outcome'].
    """
    # Drop future/unplayed matches
    completed = matches_df.dropna(subset=["home_score", "away_score"]).copy()
    completed["home_score"] = pd.to_numeric(completed["home_score"], errors="coerce")
    completed["away_score"] = pd.to_numeric(completed["away_score"], errors="coerce")
    completed = completed.dropna(subset=["home_score", "away_score"])

    # Build per-team rolling history
    history = _build_team_history(completed)
    history = _add_rolling_stats(history)
    history = _add_wc_experience(history)

    # Drop any duplicate (team, date) entries before pivoting back to match rows
    history = history.drop_duplicates(subset=["team", "date"], keep="last")

    # Pivot rolling stats back onto match rows (home = _A, away = _B)
    stat_cols = ["form_last5", "goals_scored_avg", "goals_conceded_avg", "rest_days", "wc_experience"]

    # Defining and renaming historical team stats
    home_stats = (
        history[["date", "team"] + stat_cols]
        .rename(columns={c: f"{c}_A" for c in stat_cols})
        .rename(columns={"team": "home_team"})
    )
    away_stats = (
        history[["date", "team"] + stat_cols]
        .rename(columns={c: f"{c}_B" for c in stat_cols})
        .rename(columns={"team": "away_team"})
    )

    df = completed.merge(home_stats, on=["date", "home_team"], how="left")
    df = df.merge(away_stats, on=["date", "away_team"], how="left")

    # Join FIFA rankings and compute derived feature for both teams
    rank_cols = ["team", "rank", "total_points", "confederation"]
    median_rank = pd.to_numeric(rankings_df["rank"], errors="coerce").median()

    for side, team_col in (("A", "home_team"), ("B", "away_team")):
        joined = rankings_df[rank_cols].rename(columns={
            "team": team_col,
            "rank": f"rank_{side}",
            "total_points": f"total_points_{side}",
            "confederation": f"confederation_{side}",
        })
        df = df.merge(joined, on=team_col, how="left")

    df["rank_A"] = pd.to_numeric(df["rank_A"], errors="coerce").fillna(median_rank)
    df["rank_B"] = pd.to_numeric(df["rank_B"], errors="coerce").fillna(median_rank)
    df["confederation_A"] = df["confederation_A"].fillna("")
    df["confederation_B"] = df["confederation_B"].fillna("")

    df["fifa_rank_diff"] = df["rank_A"] - df["rank_B"]

    # H2H win rate 
    df["h2h_winrate"] = _compute_h2h(df)
    
    # Knockout flag 
    df["is_knockout"] = _is_knockout_flag(df["tournament"])

    # One-hot confederation encoding
    df = encode_confederation(df)

    # Derive outcome (2=home_win, 1=draw, 0=away_win)
    df["outcome"] = (
        (df["home_score"] > df["away_score"]).astype(int) * 2
        + (df["home_score"] == df["away_score"]).astype(int)
    )

    # Final selection and cleanup
    df["year"] = df["date"].dt.year
    final_cols = FEATURE_COLS + ["year", "outcome"]
    df = df[final_cols].dropna()
    return df.reset_index(drop=True)


def get_match_features(team_a: str, team_b: str, match_date: datetime) -> np.ndarray:
    """
    Compute a feature vector for a single match ready for live model inference.
    
    Fetches current cached data (historical matches, rankings, WC results) and computes rolling
    stats up to `match_date` (exclusive), then assembles the complete feature vector in
    `FEATURE_COLS` order. Uses the same feature-engineering logic as `build_feature_matrix()` to 
    ensure training-inference consistency.

    Args:
        team_a: Home team name.
        team_b: Away team name.
        match_date: Date of the match (used to cut off rolling windows).

    Returns:
        np.ndarray: 1D numpy array aligned to `FEATURE_COLS`.
    """
    # Fetch historical and match data
    matches = data.fetch_historical_matches(n_years=25)
    rankings = data.fetch_fifa_rankings()
    wc = data.fetch_wc_results()

    # Merge completed WC2026 results into historical match pool for updated rolling stats
    if not wc.empty:
        wc_compat = wc[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
        wc_compat["tournament"] = "FIFA World Cup"
        wc_compat["neutral"] = False
        completed_wc = wc_compat.dropna(subset=["home_score", "away_score"])
        matches = pd.concat([matches, completed_wc], ignore_index=True).drop_duplicates(
            subset=["date", "home_team", "away_team"]
        )

    # Only include matches before match_date for rolling stats
    prior_matches = matches[matches["date"] < pd.Timestamp(match_date)].copy()
    history = _build_team_history(prior_matches)
    history = _add_rolling_stats(history)
    history = _add_wc_experience(history)

    # Extract latest stats
    def _latest_stats(team: str, stat_cols: list[str]) -> dict[str, float]:
        """Retrieves the latest stats for each from their history."""
        team_rows = history[history["team"] == team].sort_values("date")
        if team_rows.empty:
            return {c: float("nan") for c in stat_cols}
        last = team_rows.iloc[-1]
        return {c: float(last.get(c, float("nan"))) for c in stat_cols}

    stat_cols = ["form_last5", "goals_scored_avg", "goals_conceded_avg", "rest_days", "wc_experience"]
    stats_a = _latest_stats(team_a, stat_cols)
    stats_b = _latest_stats(team_b, stat_cols)

    # Rankings and confederation lookup
    rank_map = rankings.set_index("team")
    median_rank = pd.to_numeric(rankings["rank"], errors="coerce").median()

    def _rank(team: str) -> float:
        """Return team ranking. If team not found, return computed median rank."""
        if team in rank_map.index:
            return float(pd.to_numeric(rank_map.loc[team, "rank"], errors="coerce") or median_rank)
        return float(median_rank)

    def _conf(team: str) -> str:
        """Return team confederation. If team not found, return an empty string."""
        if team in rank_map.index:
            return str(rank_map.loc[team, "confederation"])
        return ""

    rank_a, rank_b = _rank(team_a), _rank(team_b)
    conf_a, conf_b = _conf(team_a), _conf(team_b)

    # Compute H2H win rate over prior 10 years
    cutoff = pd.Timestamp(match_date) - pd.Timedelta(days=365 * 10)
    h2h = prior_matches[
        (prior_matches["date"] >= cutoff)
        & (
            ((prior_matches["home_team"] == team_a) & (prior_matches["away_team"] == team_b))
            | ((prior_matches["home_team"] == team_b) & (prior_matches["away_team"] == team_a))
        )
    ].dropna(subset=["home_score", "away_score"])

    if h2h.empty:
        h2h_winrate = 0.5
    else:
        wins_a = (
            ((h2h["home_team"] == team_a) & (h2h["home_score"] > h2h["away_score"])).sum()
            + ((h2h["away_team"] == team_a) & (h2h["away_score"] > h2h["home_score"])).sum()
        )
        h2h_winrate = wins_a / len(h2h)

    # One-hot encode confederations
    conf_vec_a = [int(conf_a == c) for c in _CONF_LABELS]
    conf_vec_b = [int(conf_b == c) for c in _CONF_LABELS]
    
    # Assemble feature vector in FEATURE_COLS order
    vector = [
        rank_a - rank_b,                          # fifa_rank_diff
        stats_a.get("form_last5", 0.5),           # form_last5_A
        stats_b.get("form_last5", 0.5),           # form_last5_B
        stats_a.get("goals_scored_avg", 1.5),     # goals_scored_avg_A
        stats_a.get("goals_conceded_avg", 1.5),   # goals_conceded_avg_A
        stats_b.get("goals_scored_avg", 1.5),     # goals_scored_avg_B
        stats_b.get("goals_conceded_avg", 1.5),   # goals_conceded_avg_B
        stats_a.get("wc_experience", 0.0),        # wc_experience_A
        stats_b.get("wc_experience", 0.0),        # wc_experience_B
        h2h_winrate,                              # h2h_winrate
        stats_a.get("rest_days", 7.0),            # rest_days_A
        stats_b.get("rest_days", 7.0),            # rest_days_B
        0,                                        # is_knockout (caller sets if needed)
        *conf_vec_a,
        *conf_vec_b,
    ]

    return np.array(vector, dtype=float)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    # CLI interface
    parser = argparse.ArgumentParser(description="Build WC2026 feature matrix.")
    parser.add_argument("--rebuild", action="store_true", help="Delete cache and rebuild")
    args = parser.parse_args()

    # Force refresh and delete cached features if requested
    if args.rebuild:
        cache = PROCESSED_DIR / "features.parquet"
        if cache.exists():
            cache.unlink()
            print("Deleted cached features.parquet — rebuilding.")

    matches = data.fetch_historical_matches(n_years=25)
    print(f"Loaded {len(matches)} historical matches.")

    rankings = data.fetch_fifa_rankings()
    print(f"Loaded {len(rankings)} team rankings.")

    print("Building feature matrix (h2h pass may take ~30s)...")
    features = build_feature_matrix(matches, rankings)
    data.save_processed(features, "features")

    # Print summary statistics
    print(f"\nFeature Matrix: {len(features)} rows x {len(features.columns)} columns")
    print(f"Target Distribution:\n{features['outcome'].value_counts().sort_index()}")
    nan_counts = features[FEATURE_COLS].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if nan_counts.empty:
        print("NaN Counts: None — all features clean.")
    else:
        print(f"NaN Counts:\n{nan_counts}")
