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

class HealthResponse(BaseModel):
    """Liveness/readiness signal for the deployment platform and dashboard."""
    status: str = "ok"
    model_loaded: bool
    predictions_loaded: bool
    generated_at: str | None = None


class RetrainResponse(BaseModel):
    """Acknowledgement that a retrain request was accepted (or rejected)."""
    status: str  # "started" | "rejected"
    detail: str
