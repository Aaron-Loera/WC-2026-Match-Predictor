"""
Request/response Pydantic models for the WC2026 predictions API.
"""
from __future__ import annotations
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class MatchPrediction(BaseModel):
    """Win/draw/loss probabilities for a single group-stage match."""
    home: str
    away: str
    p_home_win: float = Field(ge=0.0, le=1.0)
    p_draw: float = Field(ge=0.0, le=1.0)
    p_away_win: float = Field(ge=0.0, le=1.0)


class TournamentOddsResponse(BaseModel):
    """Per-team tournament-win probabilities from the latest Monte Carlo run."""
    generated_at: str
    n_simulations: int
    tournament_odds: dict[str, float]


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class OddsHistoryEntry(BaseModel):
    """A single point-in-time snapshot of tournament odds, for tracking shifts over time."""
    generated_at: str
    tournament_odds: dict[str, float]


# ---------------------------------------------------------------------------
# Operational
# ---------------------------------------------------------------------------

class TrainingMetricsEntry(BaseModel):
    """One nightly training run: test-set metrics, Optuna hyperparameters, and pass/fail flags."""
    retrained_at: str
    test_logloss: float
    test_accuracy: float
    test_brier: float
    val_logloss: float
    best_params: dict
    n_estimators: int
    train_rows: int
    val_rows: int
    test_rows: int
    pass_logloss: bool
    pass_accuracy: bool
    pass_brier: bool


class HealthResponse(BaseModel):
    """Liveness/readiness signal for the deployment platform and dashboard."""
    status: str = "ok"
    model_loaded: bool
    predictions_loaded: bool
    generated_at: str | None = None
    last_logloss: float | None = None
    metrics_pass: bool | None = None


class RetrainResponse(BaseModel):
    """Acknowledgement that a retrain request was accepted (or rejected)."""
    status: str  # "started" | "rejected"
    detail: str
