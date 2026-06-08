"""
FastAPI startup lifecycle and dependency injection for the WC2026 predictions API

This module handles three responsibilities:

1. Startup and Shutdown (lifespan): Loads the trained XGBoost model and cached
predictions once at a process startup, avoiding the `joblib.load()` overhead on
every request. Also bootstraps the odds-history snapshot list for the `/predictions/history`
endpoint.

2. Dependency Providers: FastAPI route handlers request predictions or history
via `Depends(get_predictions)` or `Depends(get_history)`, which return
module-level cached state. This pattern ensures all routes see the same singleton
model and predictions throughout the process lifecycle.

3. Authentication & Background Jobs: Provides a bearer-token validation for `/admin/*`
routes and runs the full retraining pipeline in the background so the triggering
request returns immediately.

Module-level state is initialized by `lifespan()` at startup and mutated only by
`run_retrain_job()` after retrains complete. Routes access this state via the
dependency providers above.
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
import src.model as model
import src.simulator as simulator

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration & State
# ---------------------------------------------------------------------------

MODEL_PATH = Path("models/model.pkl")
PREDICTIONS_PATH = Path("predictions.json")
HISTORY_PATH = Path("odds_history.json")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

# Module-level shared state variables
_model = None
_predictions: dict | None = None
_history: list[dict] = []
_retrain_in_progress: bool = False


# ---------------------------------------------------------------------------
# Private Helpers: Load/Persist Predictions & History
# ---------------------------------------------------------------------------

def _load_predictions(path: Path=PREDICTIONS_PATH) -> dict:
    """
    Load `predictions.json` from disk.
    
    Raises `FileNotFoundError` if the file is missing. This blocks API startup
    to ensure fresh predictions are always available.

    Args:
        path: Path to the predictions JSON file.

    Returns:
        dict: Parsed predictions payload with keys [generated_at, n_simulations,
        tournament_odds, match_predictions].
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No predictions at {path} — run: python src/simulator.py --simulate --export"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_history(path: Path=HISTORY_PATH) -> list[dict]:
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


def _save_history(history: list[dict], path: Path=HISTORY_PATH) -> None:
    """
    Persist the odds-history snapshot list to disk as pretty-printed JSON.

    Args:
        history: The full snapshot list to write.
        path: Path to the odds-history JSON file.

    Returns:
        None:
    """
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def _record_snapshot(predictions: dict, history: list[dict]) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Lifespan: Load model & predictions once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model and predictions once at process startup (not per request).

    Avoids the `joblib.load()` overhead on every request. Also bootstraps
    or catches up `odds_history.json` so `/predictions/history` has data to serve
    from the very first run.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control returns to FastAPI, which serves requests until shutdown.
    """
    # Module level variables
    global _model, _predictions, _history

    # Startup phase
    print("Loading model and predictions at startup...")
    _model = model.load_model(MODEL_PATH)
    _predictions = _load_predictions(PREDICTIONS_PATH)
    _history = _load_history(HISTORY_PATH)

    updated = _record_snapshot(_predictions, _history)
    if updated is not _history:
        _history = updated
        _save_history(_history, HISTORY_PATH)
        print(f"odds_history.json updated — now {len(_history)} snapshot(s).")

    print("Startup complete — model and predictions loaded once.")
    
    # Hand control back to FastAPI
    yield
    # Shutdown phase
    print("API shutting down.")


# ---------------------------------------------------------------------------
# Dependency Providers
# ---------------------------------------------------------------------------

def get_predictions() -> dict:
    """
    Return the in-memory predictions payload, or raise an `HTTPException` with status
    503 if startup hasn't completed.
    
    Args:
        None:
        
    Returns:
        dict: The cached predictions payload with keys
        [generated_at, n_simulations, tournament_odds, match_predictions].
    """
    if _predictions is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Predictions not loaded yet")
    return _predictions


def get_history() -> list[dict]:
    """
    Return the in-memory odds-history snapshot list, possibly empty on first run.
    
    Args:
        None:
        
    Returns:
        list: A list of {generated_at, tournament_odds} snapshots, oldest first.
    """
    return _history


# ---------------------------------------------------------------------------
# Authentication — Bearer token for /admin/* routes
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=True)

def verify_admin_token(
    credentials: HTTPAuthorizationCredentials=Depends(_bearer_scheme),
) -> None:
    """
    Validate the `Authorization: Bearer <token>` header against `ADMIN_TOKEN`.
    
    Dependency provider for `/admin/*` routes. The `HTTPBearer(auto_error=True)` scheme
    already rejects missing or malformed headers with 403. This function additionally
    checks whether the server is misconfigured (returns 503) or the token is present but
    wrong (returns 401). Returns normally (None) only if the token matches.

    Args:
        credentials: Parsed bearer credentials, injected by FastAPI's security scheme.

    Returns:
        None: Raises HTTPException on any failure, returning normally means "authorized".
    """
    # Check if server is misconfigured
    if not ADMIN_TOKEN:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "ADMIN_TOKEN not configured on server")
    
    # Compare provided token against configured token
    if credentials.credentials != ADMIN_TOKEN:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or missing admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Retrain Job (runs in the background)
# ---------------------------------------------------------------------------

def run_retrain_job() -> None:
    """
    Retrain the model, re-export predictions, and reload all in-memory state.

    Scheduled via FastAPI `BackgroundTasks` so the triggering POST returns
    immediately rather than blocking on the multi-minute Optuna search. Runs
    the full ML pipeline in order: fetch data, build features, optimize
    hyperparameters, train model, simulate tournament. Reloads the trained
    model, predictions, and odds-history into module-level caches so subsequent
    requests see fresh data. Always clears `_retrain_in_progress` in `finally`,
    so a failed run doesn't permanently wedge the `/admin/retrain` endpoint
    behind a 409.

    Args:
        None:

    Returns:
        None:
    """
    # Module level variables
    global _model, _predictions, _history, _retrain_in_progress
    
    try:
        print("Background retrain started...")
        model.run_training_pipeline()
        simulator.export_predictions(PREDICTIONS_PATH)

        _model = model.load_model(MODEL_PATH)
        _predictions = _load_predictions(PREDICTIONS_PATH)

        updated = _record_snapshot(_predictions, _history)
        if updated is not _history:
            _history = updated
            _save_history(_history, HISTORY_PATH)

        print("Retrain complete — reloaded model, predictions, history.")
        
    except Exception as exc:
        print(f"Retrain failed: {exc!r}")
        
    finally:
        _retrain_in_progress = False
