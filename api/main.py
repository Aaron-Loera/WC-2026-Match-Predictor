"""
FastAPI application for WC2026 Match Predictor - serves XGBoost-based predictions.

This module defines the public HTTP API endpoints and applies CORS middleware for
cross-origin access from the Streamlit dashboard. All endpoints are read-only except
for the admin retrain endpoint, which requires bearer-token authentication.

Endpoints are organized into three groups:

1. Operational (`/health`): Liveness check for deployment monitoring.

2. Predictions (`/predictions/*): Serve cached match probabilities, tournament odds,
and historical odds snapshots. All data is loaded once at startup via the lifespan
context manager. Subsequent requests return the singleton in-memory cache.

3. Admin (`/admin/retrain`): Trigger a background model retrain (token-protected).
The Optuna hyperparameters search and model retraining run asynchronously so the
HTTP request returns immediately. The in-memory caches is updated when complete.

Module-level state is initialized by the `lifespan()` function from `api.dependencies`
at process startup and accessed by route handlers via dependency injection (Depends).
This pattern ensures all routes see the same singleton throughout the process lifecycle.
"""
from __future__ import annotations
from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from api.dependencies import (
    lifespan,
    get_predictions,
    get_history,
    get_metrics,
    verify_admin_token,
    run_retrain_job,
)
import api.dependencies as deps
from api.schemas import (
    MatchPrediction,
    TournamentOddsResponse,
    OddsHistoryEntry,
    HealthResponse,
    RetrainResponse,
    TrainingMetricsEntry,
)

app = FastAPI(
    title="WC2026 Match Predictor API",
    description="Serves XGBoost-based match outcome and tournament-winner predictions for FIFA World Cup 2026.",
    version="1.0.0",
    lifespan=lifespan,
)

# Public, read-only prediction data consumed by the Streamlit dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Operational
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(
    predictions: dict = Depends(get_predictions),
    metrics: list[dict] = Depends(get_metrics),
) -> HealthResponse:
    """
    Liveness check for the deployment platform.

    Returns operational status, the timestamp of currently loaded predictions, and
    a summary of the latest training run (logloss + pass/fail). Raises 503 if
    predictions haven't been loaded at startup.

    Args:
        predictions: Cached predictions payload, injected by dependency provider.
        metrics: Training-metrics history, injected by dependency provider.

    Returns:
        HealthResponse: Status, boolean flags, generated_at, last_logloss, metrics_pass.
    """
    latest = metrics[-1] if metrics else {}
    return HealthResponse(
        status="ok",
        model_loaded=deps._model is not None,
        predictions_loaded=predictions is not None,
        generated_at=predictions.get("generated_at"),
        last_logloss=latest.get("test_logloss"),
        metrics_pass=(
            all([
                latest.get("pass_logloss", True),
                latest.get("pass_accuracy", True),
                latest.get("pass_brier", True),
            ])
            if latest else None
        ),
    )


@app.get("/predictions/metrics", response_model=list[TrainingMetricsEntry], tags=["ops"])
def get_training_metrics(metrics: list[dict] = Depends(get_metrics)) -> list[dict]:
    """
    Training-run history: test metrics, Optuna hyperparameters, and pass/fail flags.

    Returns all recorded training runs oldest-first. Returns an empty list before the
    first nightly retrain has committed `models/training_metrics.json`.

    Args:
        metrics: Training-metrics history, injected by dependency provider.

    Returns:
        list: Array of TrainingMetricsEntry objects.
    """
    return metrics


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

@app.get("/predictions/matches", response_model=list[MatchPrediction], tags=["predictions"])
def get_match_predictions(predictions: dict = Depends(get_predictions)) -> list[dict]:
    """
    Win/draw/loss probabilities for all group-stage matches.

    Returns all 72 group-stage pairing with model-predicted outcome probabilities.
    No played/upcoming flags, `predictions.json` always contains all pairings regardless
    of real-world status.
    
    Args:
        predictions: Cached predictions payload from startup, injected by dependency provider.
        
    Returns:
        list: Array of `MatchPredictions` objects with home/away teams and probabilities.
    """
    return predictions["match_predictions"]


@app.get("/predictions/tournament", response_model=TournamentOddsResponse, tags=["predictions"])
def get_tournament_odds(predictions: dict = Depends(get_predictions)) -> dict:
    """
    Tournament win probability per team from the latest Monte Carlo run.

    Serves probabilities for all 48 teams plus `generated_at`/`n_simulations`
    for statistical context. Eliminated teams trend toward 0.0 as real-world
    results lock in. No explicit "eliminated" flag is included in the schema.
    
    Args:
        predictions: Cached predictions payload from startup, injected by dependency provider.
        
    Returns:
        dict: Dictionary with keys [generated_at, n_simulations, tournament_odds].
    """
    return {
        "generated_at": predictions["generated_at"],
        "n_simulations": predictions["n_simulations"],
        "tournament_odds": predictions["tournament_odds"],
    }


@app.get("/predictions/history", response_model=list[OddsHistoryEntry], tags=["predictions"])
def get_odds_history(history: list[dict] = Depends(get_history)) -> list[dict]:
    """
    Tournament-odds snapshots over time, oldest first.
    
    Returns a chronological list of odds snapshots, one per unique prediction re-export.
    Demonstrates how odds shift after each matchday as real-world results lock in and
    the simulator runs forward.
    
    Args:
        history: Cached odds-history snapshot list, injected by dependency provider.
        
    Returns:
        list: Array of `OddsHistoryEntry` objects sorted by "generated_at" in ascending order.
    """
    return history


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

@app.post(
    "/admin/retrain",
    response_model=RetrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_admin_token)],
    tags=["admin"]
)
def trigger_retrain(background_tasks: BackgroundTasks) -> RetrainResponse:
    """
    Trigger a manual model retrain (bearer-token-protected).

    Schedules the full retrain and re-export pipeline as a background task and returns
    immediately with 202 Accepted. Rejects concurrent retrains with 409 Conflict. A
    simple in-memory flag is sufficient since this API runs as a single instance.
    
    Args:
        background_tasks: FastAPI `BackgroundTasks` queue for scheduling async work.
        
    Returns:
        RetrainResponse: Confirmation that retrain was scheduled (status="started").
    """
    if deps._retrain_in_progress:
        raise HTTPException(status.HTTP_409_CONFLICT, "A retrain is already in progress")
    deps._retrain_in_progress = True
    background_tasks.add_task(run_retrain_job)
    return RetrainResponse(
        status="started",
        detail="Retrain scheduled in the background — this may take several minutes.",
    )

# Run locally:    uvicorn api.main:app --reload
# Render:         uvicorn api.main:app --host 0.0.0.0 --port $PORT   (see Dockerfile CMD)
