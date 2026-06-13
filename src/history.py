"""
Shared helpers for reading, writing, and updating `odds_history.json`.

Used by both `api/dependencies.py` (in-process snapshot bookkeeping at API startup)
and `src/simulator.py` (the `--update-history` CLI flag, run as a nightly retrain
workflow step so history persists across redeploys).
"""
from __future__ import annotations
import json
from pathlib import Path

HISTORY_PATH = Path("odds_history.json")


def load_history(path: Path = HISTORY_PATH) -> list[dict]:
    """
    Load the odds-history snapshot list from disk, or start fresh if absent/corrupt.

    Returns an empty list on first-ever run or if the file is malformed.

    Args:
        path: Path to the odds-history JSON file.

    Returns:
        list: A list of {generated_at, tournament_odds} snapshots, oldest first.
    """
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"odds_history.json unreadable ({exc!r}) — starting a fresh history.")
        return []


def save_history(history: list[dict], path: Path = HISTORY_PATH) -> None:
    """
    Persist the odds-history snapshot list to disk as pretty-printed JSON.

    Args:
        history: The full snapshot list to write.
        path: Path to the odds-history JSON file.

    Returns:
        None:
    """
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def record_snapshot(predictions: dict, history: list[dict]) -> list[dict]:
    """
    Append a deduplicated {generated_at, tournament_odds} snapshot if `predictions` is new.

    Compares the "generated_at" timestamp in the current predictions against the most recent
    snapshot in history.

    Args:
        predictions: The freshly loaded predictions payload.
        history: The current snapshot list (oldest first).

    Returns:
        list: `history` unchanged if nothing new; otherwise a new list with one snapshot appended.
    """
    new_ts = predictions.get("generated_at")

    # Return the same history if no changes detected
    if new_ts is None or (history and history[-1].get("generated_at") == new_ts):
        return history

    # Create and return new snapshot
    snapshot = {"generated_at": new_ts, "tournament_odds": predictions["tournament_odds"]}
    return history + [snapshot]
