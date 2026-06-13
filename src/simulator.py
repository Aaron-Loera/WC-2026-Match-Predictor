from __future__ import annotations
import json
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from time import perf_counter
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
import src.data as data
from src.features import (
    FEATURE_COLS,
    _CONF_LABELS,
    _build_team_history,
    _add_rolling_stats,
    _add_wc_experience,
)
from src.model import load_model, predict_proba
import src.history as history

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREDICTIONS_PATH = Path("predictions.json")
HISTORY_PATH = Path("odds_history.json")
N_SIMULATIONS = 10_000
_CACHE_DATE = datetime(2026, 6, 11)  # feature lookback cutoff — tournament start

# WC2026 groups from the official draw
WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador",],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
}

# Official WC2026 R32 bracket (matches 73-88, source: Wikipedia knockout stage article).
# Adjacent pairs feed the same R16 match (M73/M74 -> R16, M75/M76 -> R16, ...).
# Third-place slots (T1-T8) are assigned rank-order to their W-vs-T slots; the full
# 495-scenario seeding matrix (FIFA Annex C) is not implemented -- acceptable approximation.
_R32_TEMPLATE: list[tuple[str, str]] = [
    ("RA", "RB"),  # M73 -- runner-up A vs runner-up B
    ("WE", "T1"),  # M74 -- winner E vs best 3rd
    ("WF", "RC"),  # M75 -- winner F vs runner-up C
    ("WC", "RF"),  # M76 -- winner C vs runner-up F
    ("WI", "T2"),  # M77 -- winner I vs 2nd-best 3rd
    ("RE", "RI"),  # M78 -- runner-up E vs runner-up I
    ("WA", "T3"),  # M79 -- winner A vs 3rd-best 3rd
    ("WL", "T4"),  # M80 -- winner L vs 4th-best 3rd
    ("WD", "T5"),  # M81 -- winner D vs 5th-best 3rd
    ("WG", "T6"),  # M82 -- winner G vs 6th-best 3rd
    ("RK", "RL"),  # M83 -- runner-up K vs runner-up L
    ("WH", "RJ"),  # M84 -- winner H vs runner-up J
    ("WB", "T7"),  # M85 -- winner B vs 7th-best 3rd
    ("WJ", "RH"),  # M86 -- winner J vs runner-up H
    ("WK", "T8"),  # M87 -- winner K vs 8th-best 3rd
    ("RD", "RG"),  # M88 -- runner-up D vs runner-up G
]

# Module-level locked results (populated by lock_result())
_locked_results: dict[tuple[str, str], tuple[int, int]] = {}


# ---------------------------------------------------------------------------
# Private Helpers: Data Preparation
# ---------------------------------------------------------------------------

def _load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict], dict]:
    """
    Load raw data and extract per-team statistics and head-to-head records for probability computation.
    
    Fetches historical matches (25 years), FIFA rankings, and any completed WC2026 results. Merges
    completed WC2026 matches into the historical pool to ensure rollings stats are reflect up-to-date
    team form. Constructs a normalized H2H win-count lookup over the prior 10 years.
    
    Args:
        None:
        
    Returns:
        tuple: A 4-tuple containing:
            - prior (pd.DataFrame): All completed matches before tournament start.
            - rankings (pd.DataFrame): Team rankings.
            - team_stats (dict): Per-team rollings stats keys by team name.
            - h2h_lookup (dict): Head-to-head record.
    
    Load raw data once and build per-team stats + H2H lookup for batch feature construction.
    """
    # Fetch historical and current data
    matches = data.fetch_historical_matches(n_years=25)
    rankings = data.fetch_fifa_rankings()
    wc = data.fetch_wc_results()

    # Merge completed WC2026 results into historical pool
    if not wc.empty:
        wc_compat = wc[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
        wc_compat["tournament"] = "FIFA World Cup"
        wc_compat["neutral"] = False
        completed = wc_compat.dropna(subset=["home_score", "away_score"])
        matches = pd.concat([matches, completed], ignore_index=True).drop_duplicates(
            subset=["date", "home_team", "away_team"]
        )

    # Enrich frame with rolling stats
    prior = matches[matches["date"] < pd.Timestamp(_CACHE_DATE)].copy()
    history = _build_team_history(prior)
    history = _add_rolling_stats(history)
    history = _add_wc_experience(history)

    # Extract most-recent rolling stats per WC team
    stat_cols = ["form_last5", "goals_scored_avg", "goals_conceded_avg", "rest_days", "wc_experience"]
    all_teams = [t for g in WC2026_GROUPS.values() for t in g]
    team_stats: dict[str, dict] = {}
    for team in all_teams:
        rows = history[history["team"] == team].sort_values("date")
        if rows.empty:
            team_stats[team] = {c: float("nan") for c in stat_cols}
        else:
            last = rows.iloc[-1]
            team_stats[team] = {c: float(last.get(c, float("nan"))) for c in stat_cols}

    # Build H2H win-count lookup over the last 10 years
    cutoff = pd.Timestamp(_CACHE_DATE) - pd.Timedelta(days=365 * 10)
    h2h_df = prior[prior["date"] >= cutoff].dropna(subset=["home_score", "away_score"])
    h2h_raw: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for _, row in h2h_df.iterrows():
        h, a = str(row["home_team"]), str(row["away_team"])
        key: tuple[str, str] = (min(h, a), max(h, a))
        first_won = int(
            (h == key[0] and row["home_score"] > row["away_score"])
            or (a == key[0] and row["away_score"] > row["home_score"])
        )
        h2h_raw[key][0] += first_won
        h2h_raw[key][1] += 1
    h2h_lookup = {k: (v[0], v[1]) for k, v in h2h_raw.items()}

    return prior, rankings, team_stats, h2h_lookup


def _build_proba_cache(
    model,
    prior: pd.DataFrame,
    rankings: pd.DataFrame,
    team_stats: dict[str, dict],
    h2h_lookup: dict[tuple[str, str], tuple[int, int]],
) -> dict[tuple[str, str], np.ndarray]:
    """   
    Pre-compute match outcome probabilities for all ordered team pairs via batch model inference.
    
    Construct a feature matrix for all ordered (team_a, team_b) matchups using rankings,
    rollings stats, head-to-head records, and confederation memberships. Runs the XGBoost model
    once on the full matrix, then caches the resulting probabilities for O(1) lookup during
    Monte Carlo simulation. Matches not in cache default to uniform [1/3, 1/3, 1/3]
    
    Args:
        model: Fitted XGBoost classifier with `predict_proba()` method.
        prior: Historical match data (unused but kept for future context).
        rankings: Team rankings with columns [rank, team, total_points, confederation].
        team_stats: Per-team rollings stats from `_load_and_prepare()`.
        h2h_lookup: H2H record lookup from `_load_and_prepare()`.
        
    Returns:
        dict: Probability cache containing [p_away_win, p_draw, p_home_win].
    """
    all_teams = [t for g in WC2026_GROUPS.values() for t in g]
    rank_map = rankings.set_index("team")
    median_rank = float(pd.to_numeric(rankings["rank"], errors="coerce").median())

    def _rank(team: str) -> float:
        """Return a team's FIFA rank or the median if it's missing."""
        if team in rank_map.index:
            v = pd.to_numeric(rank_map.loc[team, "rank"], errors="coerce")
            return float(v) if not pd.isna(v) else median_rank
        return median_rank

    def _conf_vec(team: str) -> list[int]:
        """Return a one-hot encoded confederation vector."""
        conf = str(rank_map.loc[team, "confederation"]) if team in rank_map.index else ""
        return [int(conf == c) for c in _CONF_LABELS]

    def _h2h_rate(team_a: str, team_b: str) -> float:
        """Returns `team_a`'s win rate versus `team_b`."""
        key: tuple[str, str] = (min(team_a, team_b), max(team_a, team_b))
        entry = h2h_lookup.get(key)
        if entry is None or entry[1] == 0:
            return 0.5
        rate = entry[0] / entry[1]
        return rate if team_a <= team_b else 1.0 - rate

    def _stat(stats: dict, col: str, default: float) -> float:
        """Safely extract a stat returning a default if it's missing or NaN."""
        v = stats.get(col, float("nan"))
        return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    # Calculating expected speed of ordered pairs
    t0 = perf_counter()
    n_pairs = len(all_teams) * (len(all_teams) - 1)
    print(f"Pre-computing probabilities for {n_pairs:,} team pairs...")

    # Generating ordered pairs
    pairs = list(itertools.permutations(all_teams, 2))
    X = np.zeros((len(pairs), len(FEATURE_COLS)), dtype=np.float32)


    for i, (team_a, team_b) in enumerate(pairs):
        # Get rollings stats and confederation vectors
        sa, sb = team_stats[team_a], team_stats[team_b]
        cv_a = _conf_vec(team_a)
        cv_b = _conf_vec(team_b)
        
        # Build feature vectors
        X[i] = [
            _rank(team_a) - _rank(team_b),       # fifa_rank_diff
            _stat(sa, "form_last5", 0.5),        # form_last5_A
            _stat(sb, "form_last5", 0.5),        # form_last5_B
            _stat(sa, "goals_scored_avg", 1.5),  # goals_scored_avg_A
            _stat(sa, "goals_conceded_avg", 1.5),# goals_conceded_avg_A
            _stat(sb, "goals_scored_avg", 1.5),  # goals_scored_avg_B
            _stat(sb, "goals_conceded_avg", 1.5),# goals_conceded_avg_B
            _stat(sa, "wc_experience", 0.0),     # wc_experience_A
            _stat(sb, "wc_experience", 0.0),     # wc_experience_B
            _h2h_rate(team_a, team_b),           # h2h_winrate
            _stat(sa, "rest_days", 7.0),         # rest_days_A
            _stat(sb, "rest_days", 7.0),         # rest_days_B
            0,                                   # is_knockout (group stage default)
            *cv_a,                               # conf_*_A (6 one-hot)
            *cv_b,                               # conf_*_B (6 one-hot)
        ]

    # Run XGBoost inference on all pairs
    proba_matrix = predict_proba(model, X)
    
    # Create and return cache 
    cache = {pair: proba_matrix[i] for i, pair in enumerate(pairs)}
    print(f"Cache ready in {perf_counter() - t0:.1f}s — {len(cache):,} matchups.")
    return cache


def _prepare_simulation() -> tuple[dict[tuple[str, str], np.ndarray], dict[str, float], int]:
    """
    Prepare a one-time setup: load a model, fetch the data, compute team stats,
    and cache match probabilities.
    
    Prints a warning statement for `cache_misses` if team pairs are absent from `proba_cache`
    indicating a team name mismatch causing a uniform-probability fallback for that pair.

    Args:
        None:

    Returns:
        tuple: A 3-tuple containing (proba_cache, rank_lookups, cache_misses).
    """
    # Load model and data
    model = load_model()
    prior, rankings, team_stats, h2h_lookup = _load_and_prepare()

    # Create FIFA rank lookup
    rank_lookup = {
        str(row["team"]): float(pd.to_numeric(row["rank"], errors="coerce") or 999)
        for _, row in rankings.iterrows()
    }

    # Compute match probabilities
    proba_cache = _build_proba_cache(model, prior, rankings, team_stats, h2h_lookup)

    # Count how many team pairs are absent
    all_teams = [t for group in WC2026_GROUPS.values() for t in group]
    cache_misses = sum(
        1 for a, b in itertools.permutations(all_teams, 2)
        if (a, b) not in proba_cache
    )
    if cache_misses:
        print(f"WARNING: {cache_misses} team-pair(s) missing from proba_cache — uniform fallback active.")

    return proba_cache, rank_lookup, cache_misses


# ---------------------------------------------------------------------------
# Private Helpers: Simulation Logic
# ---------------------------------------------------------------------------

def _get_outcome(
    home: str,
    away: str,
    proba_cache: dict[tuple[str, str], np.ndarray],
) -> int:
    """
    Return sampled outcome: 0=away win, 1=draw, 2=home win.
    
    Sample a group-stage match outcome, or return a locked result if available.
    
    Check if (home, away) matchup has a confirmed real-world result via `lock_result()`.
    If so, compare scores and returns the outcome from the home team's perspective.
    Otherwise, looks up the pre-computed model probabilities in the cache and samples an
    outcome (0=away_win, 1=draw, 2=home_win).
    
    Args:
        home: Home team name.
        away: Away team name.
        proba_cache: Pre-computed match probabilities from `_build_proba_cache()`.
        
    Returns:
        int: Match outcome from home team's perspective: 0=away_win, 1=draw, 2=home_win.
    """
    # Check if matchup has a locked result
    if (home, away) in _locked_results:
        hs, as_ = _locked_results[(home, away)]
        return 2 if hs > as_ else (1 if hs == as_ else 0)
    
    # Search for reversed matchup
    if (away, home) in _locked_results:
        as_, hs = _locked_results[(away, home)]
        return 2 if hs > as_ else (1 if hs == as_ else 0)
    
    # Lookup model's predicted probabilities
    proba = proba_cache.get((home, away))
    
    if proba is None:
        return int(np.random.choice(3))
    return int(np.random.choice(3, p=proba / proba.sum())) 


def _sample_knockout_winner(
    home: str,
    away: str,
    proba_cache: dict[tuple[str, str], np.ndarray],
) -> str:
    """
    Sample a knockout winner, or return an available locked real-world result.
    
    Check if the (home, away) knockout match has a confirmed real-world result
    via `lock_results()`. If so, return the winning team with draws treated as 
    home advance via penalty parity. Otherwise, look up the pre-computed model
    probabilities and sample a winner. If the matchup is not in cache, or
    probabilities are near zero, default to a 50/50 coin flip.
    
    Args:
        home: Home team name.
        away: Away team name.
        proba_cached: Pre-computed match probabilities from `_build_proba_cache()`.
        
    Returns:
        str: The advancing team's name.
    """
    # Check if a this match has a locked result
    if (home, away) in _locked_results:
        hs, as_ = _locked_results[(home, away)]
        return home if hs >= as_ else away  # treat draws as home win
    
    # Search for reversed match
    if (away, home) in _locked_results:
        as_, hs = _locked_results[(away, home)]
        return home if hs >= as_ else away
    
    # Lookup model's predicted probabilities
    proba = proba_cache.get((home, away))
    
    # Default 50/50 choice
    if proba is None:
        return home if np.random.random() < 0.5 else away
    
    # Extract and normalize probabilities
    p_win, p_loss = float(proba[2]), float(proba[0])
    denom = p_win + p_loss
    
    if denom < 1e-9:
        return home if np.random.random() < 0.5 else away
    return home if np.random.random() < (p_win / denom) else away


def _run_monte_carlo(
    n: int,
    proba_cache: dict[tuple[str, str], np.ndarray],
    rank_lookup: dict[str, float],
) -> dict[str, float]:
    """    
    Run `n` Monte Carlo tournament simulations and compute tournament win probabilities
    for all teams. 
    
    Tallies how many times each team wins, then converts counts to win
    probabilities. Execute `n` independent tournament simulations, each consisting of:
    1. Simulating the group stage with all 72 group matches.
    2. Selecting the 8 best third-place qualifiers.
    3. Constructing the R32 knockout bracket.
    4. Simulating all knockout rounds to determine a champion.
    
    Args:
        n: Number of Monte Carlo simulations to run.
        proba_cache: Pre-computed match probabilities from `_prepare_simulation()`.
        rank_lookup: Team FIFA rank lookups from `_prepare_simulation()`.
        
    Returns:
        dict: Tournament win probabilities.
    """
    all_teams = [t for g in WC2026_GROUPS.values() for t in g]
    win_counts: dict[str, int] = defaultdict(int)
    t0 = perf_counter()
    print(f"Running {n:,} Monte Carlo simulations...")
    for _ in range(n):
        standings, points = simulate_group_stage(proba_cache, rank_lookup)
        third = _select_third_place_qualifiers(standings, points, rank_lookup)
        bracket = _build_r32_bracket(standings, third)
        champion = simulate_knockout(bracket, proba_cache)
        win_counts[champion] += 1
    print(f"Completed {n:,} simulations in {perf_counter() - t0:.1f}s")
    return {team: round(win_counts[team] / n, 4) for team in all_teams}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_group_stage(
    proba_cache: dict[tuple[str, str], np.ndarray],
    rank_lookup: dict[str, float],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """
    Simulate group stage for one run. Play all 72 matches, award points, and
    rank teams within each group.
    
    Play all 6 matches in each of the 12 groups by sampling outcomes from the
    probability cache. Assign 3 points for a win, 1 point for a draw, and 0 for
    a loss. Rank the 4 teams in each group by total points (descending), with
    FIFA rank as a tiebreaker (ascending).

    Args:
        proba_cache: Pre-computed match probabilities from `_build_proba_cache()`.
        rank_lookup: Team FIFA rank lookup from `_prepare_simulation()`.
        
    Returns:
        tuple: A two-tuple containing (standings, points_by_team)
    """
    standings: dict[str, list[str]] = {}
    points_by_team: dict[str, int] = {}

    for group, teams in WC2026_GROUPS.items():
        points: dict[str, int] = {t: 0 for t in teams}
        # Generate all 6 matches for the group
        for home, away in itertools.combinations(teams, 2):
            outcome = _get_outcome(home, away, proba_cache)
            if outcome == 2:
                points[home] += 3
            elif outcome == 1:
                points[home] += 1
                points[away] += 1
            else:
                points[away] += 3
        # Sort the group (4 teams) by points and then FIFA rank
        sorted_teams = sorted(teams, key=lambda t: (-points[t], rank_lookup.get(t, 999)))
        
        standings[group] = sorted_teams
        points_by_team.update(points)

    return standings, points_by_team


def simulate_knockout(
    bracket: list[tuple[str, str]],
    proba_cache: dict[tuple[str, str], np.ndarray],
) -> str:
    """
    Simulate all knockout rounds (R32 -> R16 -> QF -> SF -> Final) and return the tournament
    champion.
    
    Iteratively play knockout matches, advancing winners until only one team remains. Each
    round samples match outcomes using `_sample_knockout_winner()`. Winners are paired consecutively
    for the next round.
    
    Args:
        bracket: Initial R32 bracket (16 (home, away) team pairs).
        proba_cache: Pre-computed match probabilities from `_build_proba_cache()`.
        
    Returns:
        str: The team's name of the tournament champion.
    """
    current_round = bracket
    # Keep playing rounds until one match remains
    while len(current_round) > 1:
        winners = [_sample_knockout_winner(h, a, proba_cache) for h, a in current_round]
        current_round = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
        
    home, away = current_round[0]
    return _sample_knockout_winner(home, away, proba_cache)


def simulate_tournament(n: int=N_SIMULATIONS) -> dict[str, float]:
    """
    Run `n` Monte Carlo simulations and return win probabilities for all 48 teams.
    
    Args:
        n: Number of Monte Carlo simulations to run.
        
    Returns:
        dict: Tournament win probabilities.
    """
    proba_cache, rank_lookup, _cache_misses = _prepare_simulation()
    return _run_monte_carlo(n, proba_cache, rank_lookup)


def lock_result(team_a: str, team_b: str, score_a: int, score_b: int) -> None:
    """
    Lock a completed real-world match result so the simulator uses it instead of sampling
    from pre-computed probabilities.
    
    Args:
        team_a: Team a's name.
        team_b: Team b's name.
        score_a: Team a's score.
        score_b: Team b's score.
        
    Returns:
        None:
    """
    _locked_results[(team_a, team_b)] = (score_a, score_b)


def export_predictions(output_path: Path | str=PREDICTIONS_PATH) -> None:
    """
    Run the full simulation pipeline and write tournament odds and match predictions to JSON.
    
    Orchestrate the end-to-end workflow:
    1. Load the model.
    2. Fetch and process data.
    3. Pre-compute match outcome probabilities for all team pairs.
    4. Run Monte Carlo tournament simulations.
    5. Extract match-level predictions for all 72-matches.
    6. Generate and serialize a JSON payload.
    
    Args:
        output_path: Path where `predictions.json` will be written (defaults to `PREDICTIONS_PATH`).
        
    Returns:
        None:
    """
    output_path = Path(output_path)
    
    # Prepare and run simulation
    proba_cache, rank_lookup, cache_misses = _prepare_simulation()
    tournament_odds = _run_monte_carlo(N_SIMULATIONS, proba_cache, rank_lookup)

    # Get match predictions for all 72 matches
    match_predictions = []
    for teams in WC2026_GROUPS.values():
        for home, away in itertools.combinations(teams, 2):
            p = proba_cache.get((home, away), np.full(3, 1 / 3))
            match_predictions.append({
                "home": home,
                "away": away,
                "p_home_win": round(float(p[2]), 4),
                "p_draw": round(float(p[1]), 4),
                "p_away_win": round(float(p[0]), 4),
            })

    # Construct output JSON structure
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_simulations": N_SIMULATIONS,
        "cache_misses": cache_misses,
        "tournament_odds": tournament_odds,
        "match_predictions": match_predictions,
    }
    
    # Save serialized payload
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Predictions written to {output_path}")


def update_history(
    predictions_path: Path | str=PREDICTIONS_PATH,
    history_path: Path | str=HISTORY_PATH,
) -> None:
    """
    Append a deduplicated odds snapshot from `predictions.json` to `odds_history.json`.

    Reads the freshly exported predictions payload, compares its "generated_at" timestamp
    against the most recent entry in `odds_history.json`, and appends a new
    {generated_at, tournament_odds} snapshot if it's new. No operating takes place if the
    timestamp already matches the latest history entry.

    Args:
        predictions_path: Path to the predictions JSON file to read from.
        history_path: Path to the odds-history JSON file to update.

    Returns:
        None:
    """
    predictions_path = Path(predictions_path)
    history_path = Path(history_path)

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    current = history.load_history(history_path)
    updated = history.record_snapshot(predictions, current)

    if updated is not current:
        history.save_history(updated, history_path)
        print(f"odds_history.json updated — now {len(updated)} snapshot(s).")
    else:
        print("odds_history.json unchanged — snapshot already recorded.")


# ---------------------------------------------------------------------------
# Private Helpers: Bracket Construction
# ---------------------------------------------------------------------------

def _select_third_place_qualifiers(
    standings: dict[str, list[str]],
    points_by_team: dict[str, int],
    rank_lookup: dict[str, float],
) -> list[str]:
    """   
    Return the 8 best third-place teams from 12 groups to advance to the Round of 32.
    
    Extract the third-place finisher from each of the 12 groups, then rank them by total
    group points (descending) with FIFA rank as a tiebreaker (ascending). Return the top
    8 teams.
    
    Args:
        standings: Dictionary mapping group letter (A-L) to [winner, runner-up, third, fourth].
        points_by_team: Dictionary mapping team name to total group-stage points.
        rank_lookup: Dictionary mapping team name to FIFA ranking.
        
    Returns:
        list: Ordered list of 8 team names that advance to the Round of 32, ranked by strength.
    """
    # Extract 3rd place teams from each group
    third_place = [standings[g][2] for g in WC2026_GROUPS]
    
    # Sort third-place teams by group points (desc) and FIFA rank (asc)
    ranked = sorted(
        third_place,
        key=lambda t: (-points_by_team.get(t, 0), rank_lookup.get(t, 999)),
    )
    return ranked[:8]


def _build_r32_bracket(
    standings: dict[str, list[str]],
    third_qualifiers: list[str],
) -> list[tuple[str, str]]:
    """
    Resolve the R32 bracket template into 16 concrete (home, away) team pairs.
    
    Maps group-stage outcomes (winners, runner-ups, qualifier third-place teams) to
    the abstract bracket template, `R32_TEMPLATE`, to produce 16 Round-of-32 matches.
    
    Args:
        standings: Dictionary mapping group letter (A-L) to [winner, runner-up, third, fourth].
        third_qualifiers: Ordered list of 8 third-place teams from `_select_third_place_qualifiers()`.
        
    Returns:
        list: List of 16 (home, away) tuples representing the Round-of-32 bracket matches.
    """
    slot_map: dict[str, str] = {}
    for g, teams in standings.items():
        slot_map[f"W{g}"] = teams[0] # group winners from each group
        slot_map[f"R{g}"] = teams[1] # runner-ups from each group
    for i, team in enumerate(third_qualifiers, 1):
        slot_map[f"T{i}"] = team # third-place qualifiers in rank order
    return [(slot_map[h], slot_map[a]) for h, a in _R32_TEMPLATE]


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Setup command line argument parser
    parser = argparse.ArgumentParser(description="Run WC2026 Monte Carlo simulator.")
    parser.add_argument("--simulate", action="store_true", help="Run simulation and print top-10 odds")
    parser.add_argument("--export", action="store_true", help="Write predictions.json (requires --simulate)")
    parser.add_argument("--update-history", action="store_true",
                         help="Append a snapshot to odds_history.json from predictions.json (run after --export)")
    args = parser.parse_args()

    # Run the complete pipeline
    if args.simulate and args.export:
        export_predictions()
    # Run simulation only
    elif args.simulate:
        odds = simulate_tournament()
        print("\n--- Top-10 Tournament Win Probabilities ---")
        for team, p in sorted(odds.items(), key=lambda x: -x[1])[:10]:
            print(f"  {team:<30} {p:.3f}")
    elif not args.update_history:
        parser.print_help()

    if args.update_history:
        update_history()
