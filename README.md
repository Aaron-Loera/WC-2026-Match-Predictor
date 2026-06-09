# WC 2026 Match Predictor

> XGBoost-powered FIFA World Cup 2026 match outcome predictor and tournament simulator — served via a live FastAPI backend and Streamlit dashboard, retrained nightly via GitHub Actions.

[![Nightly Retrain](https://github.com/Aaron-Loera/WC-2026-Match-Predictor/actions/workflows/retrain.yml/badge.svg)](https://github.com/Aaron-Loera/WC-2026-Match-Predictor/actions/workflows/retrain.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Given any two of the 48 qualified teams, the model returns calibrated win/draw/loss probabilities. A Monte Carlo simulator runs 10,000 bracket simulations to produce tournament-win odds for every team — updated after each matchday.

---

## Live Demo

| Service | URL |
|---------|-----|
| Dashboard | https://wc-2026-match-predictor.streamlit.app/|
| API docs | https://wc-2026-match-predictor-1.onrender.com/docs|

```bash
# Quick health check
curl https://<your-render-url>/health
```

> Predictions retrained nightly at 06:00 UTC and reflect all completed results.

---

## How It Works

```
football-data.org / FIFA Rankings CSV
          │
          ▼
    src/data.py        fetch + cache raw data (staleness-aware)
          │
          ▼
  src/features.py      28-feature matrix; rolling stats use .shift(1) — no leakage
          │
          ▼
   src/model.py        XGBoost multi:softprob, Optuna HPO (30 trials)
          │
          ▼
 src/simulator.py      10,000-run Monte Carlo bracket (WC2026 format)
          │
          ▼
  predictions.json ──► FastAPI (Render)
                             │
                             ▼
                    Streamlit Dashboard (Cloud)
```

| Module | Role |
|--------|------|
| `src/data.py` | Fetches historical results and FIFA rankings; caches locally with staleness thresholds |
| `src/features.py` | Builds the per-match feature matrix used for both training and live inference |
| `src/model.py` | Trains and evaluates the XGBoost classifier; writes `models/model.pkl` |
| `src/simulator.py` | Simulates the full WC2026 bracket; locks completed results; writes `predictions.json` |
| `api/main.py` | FastAPI app — loads model once at startup, serves predictions via REST |
| `dashboard/app.py` | Dark-themed Streamlit dashboard — fetches from API, no ML imports |

---

## Model Card

| Property | Value |
|----------|-------|
| Algorithm | XGBoost `multi:softprob` (Win / Draw / Loss → +1 / 0 / −1) |
| Features | 28 — FIFA rank diff, rolling form (last 5), goals scored/conceded avg, H2H win rate, WC experience, rest days, is_knockout, confederation (one-hot) |
| Train split | Temporal: pre-2022 train / 2022 WC validation / 2023–2025 test |
| HPO | Optuna TPE sampler, 30 trials, minimising validation log-loss |
| Targets | Log-loss < 1.0 · Top-1 accuracy > 52% · Brier score < 0.22 |
| Serialisation | `joblib` → `models/model.pkl` |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check + model-loaded status |
| `GET` | `/predictions/matches` | Win/draw/loss probabilities for all group-stage matches |
| `GET` | `/predictions/tournament` | Monte Carlo tournament-win odds for all 48 teams |
| `GET` | `/predictions/history` | Historical odds snapshots (odds tracker) |
| `POST` | `/admin/retrain` | Trigger a manual retrain (bearer-token protected) |

Full interactive docs at `/docs` (Swagger UI) and `/redoc`.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Aaron-Loera/WC-2026-Match-Predictor.git
cd WC-2026-Match-Predictor
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add your FOOTBALL_DATA_API_KEY and ADMIN_TOKEN

# 3. Run the full pipeline
python src/data.py --update          # fetch latest match data
python src/features.py               # build feature matrix
python src/model.py --retrain        # train model  →  models/model.pkl
python src/simulator.py --simulate --export  # simulate  →  predictions.json

# 4. Start the API and dashboard
uvicorn api.main:app --reload        # http://localhost:8000
streamlit run dashboard/app.py       # http://localhost:8501
```

A `.devcontainer` is included for VS Code Dev Containers — open the repo in a container for a fully configured environment with no local setup.

---

## Deployment

| Layer | Platform | Trigger |
|-------|----------|---------|
| API | Render free tier (Docker) | Auto-redeploys on push to `main` |
| Dashboard | Streamlit Community Cloud | Auto-redeploys on push to `main` |
| Retraining | GitHub Actions cron `0 6 * * *` | Nightly; also manually dispatchable |

The nightly pipeline: fetches data → rebuilds features → retrains model → simulates bracket → commits `models/model.pkl` + `predictions.json` → Render redeploys the API automatically.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FOOTBALL_DATA_API_KEY` | Yes | — | Free API key from football-data.org |
| `ADMIN_TOKEN` | Yes | — | Bearer token for `POST /admin/retrain` |
| `API_BASE_URL` | Dashboard only | `http://localhost:8000` | API base URL for the dashboard |
| `DASHBOARD_CACHE_TTL` | No | `3600` | `st.cache_data` TTL in seconds (use `300` on active matchdays) |

---

## Repository Layout

```
wc2026-predictor/
├── src/                    # ML pipeline (data → features → model → simulator)
├── api/                    # FastAPI app + Pydantic schemas + dependency injection
├── dashboard/              # Streamlit dashboard (app.py = production, app_demo.py = reference)
├── data/
│   ├── raw/                # FIFA rankings CSV + historical results parquet
│   └── processed/          # Feature matrix (features.parquet)
├── models/
│   └── model.pkl           # Trained XGBoost model (joblib, ~524 KB)
├── .github/workflows/
│   └── retrain.yml         # Nightly retraining cron
├── .streamlit/
│   └── config.toml         # Dark theme (charcoal + pitch-green palette)
├── predictions.json        # Latest simulation output (API contract)
├── odds_history.json       # Historical odds snapshots
├── Dockerfile              # Lean Python 3.11-slim image for Render
└── requirements*.txt       # Core / API / dashboard dependency splits
```

---

## Development Notes

- **Temporal splits only** — never use random train/test splits on this time-series dataset.
- **No data leakage** — all rolling features use `.shift(1)`; the same `build_feature_matrix()` call is safe for training and live inference.
- **Single model load** — `api/dependencies.py` loads `models/model.pkl` once at FastAPI startup via a lifespan context manager. Do not load per-request.
- **Simulator fidelity** — N=10,000 runs gives ~0.3% probability resolution. Do not reduce without good reason.
- **predictions.json is the API contract** — schema must always include `generated_at`, `n_simulations`, `tournament_odds`, and `match_predictions`.

---

## License

MIT
