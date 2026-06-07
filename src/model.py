from __future__ import annotations
import argparse
import sys
import joblib
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent))
import src.data as data
from src.features import FEATURE_COLS, build_feature_matrix

MODEL_PATH = Path("models/model.pkl")

_TRAIN_BEFORE_YEAR = 2022
_VAL_YEAR = 2022
_TEST_FROM_YEAR = 2023
_OPTUNA_N_TRIALS = 30
_EARLY_STOP_ROUNDS = 30

_DEFAULT_PARAMS: dict = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "verbosity": 0,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _load_or_rebuild_features() -> pd.DataFrame:
    """
    Load the cached feature matrix, or rebuild it from raw data if missing or invalid.
    
    Attempts to load the pre-built feature matrix from the processed directory. If the cache exists
    and contains the required `year` column, the matrix is returned. Otherwise, it's rebuilt.
    
    Args:
        None:
        
    Returns:
        pd.DataFrame: A feature matrix with columns `features.FEATURE_COLS` + `year` + `outcome`.
    """
    try:
        # Attempts to read pre-built feature matrix
        df = data.load_processed("features")
        if "year" in df.columns:
            return df
        print("Cached 'features.parquet' missing 'year' column — rebuilding.")
    except FileNotFoundError:
        # If cached file is malformed, rebuild matrix
        print("features.parquet not found — building from raw data.")

    # Fetch raw data
    matches = data.fetch_historical_matches(n_years=25)
    rankings = data.fetch_fifa_rankings()
    
    # Build and save feature matrix
    print("Building feature matrix (h2h pass may take ~30s)...")
    df = build_feature_matrix(matches, rankings)
    data.save_processed(df, "features")
    return df


def _temporal_split(
    df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split historical match data into train/val/test sets using temporal boundaries.
    
    Partition the feature DataFrame into three time-ordered subsets to avoid data leakage.
    
    Args:
        df: A DataFrame with columns `year`, `outcome`, and `features.FEATURE_COLS`.
        
    Returns:
        tuple: A 6-tuple of NumPy arrays: (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    assert "year" in df.columns, "DataFrame must contain a 'year' column for temporal splitting."
    
    # Create split masks
    train_mask = df["year"] < _TRAIN_BEFORE_YEAR
    val_mask = df["year"] == _VAL_YEAR
    test_mask = df["year"] >= _TEST_FROM_YEAR

    def _split(mask: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Extracts features and labels for a given mask."""
        X = df.loc[mask, FEATURE_COLS].to_numpy(dtype=np.float32)
        y = df.loc[mask, "outcome"].to_numpy(dtype=np.int32)
        return X, y

    # Define splits with corresponding features and labels
    X_train, y_train = _split(train_mask)
    X_val, y_val = _split(val_mask)
    X_test, y_test = _split(test_mask)

    print(
        f"Split Sizes — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def _run_optuna_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> tuple[dict, int]:
    """
    Run Bayesian hyperparameter search using Optuna to minimize validation log-loss.
    
    Executes a 30-trial Optuna optimization loop with TPE sampler to find the XGBoost hyperparameter
    combination that minimizes validation log-loss. Each trial trains a model on the training set,
    evaluates on the validation set, and records the iteration at which validation loss was lowest.
    
    Args:
        X_train: Training features.
        y_train: Training labels/outcomes.
        X_val: Validation features.
        y_val: Validation labels/outcomes.
        
    Returns:
        tuple: (best_params, best_n_estimators) where `best_params` is a dict of tuned
        hyperparameters and `best_n_estimators` is the optimal number of boosting rounds.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Assign weight distribution for labels
    w_train = compute_sample_weight("balanced", y_train)

    def objective(trial: optuna.Trial) -> float:
        """Suggests hyperparameter values and returns validation log-loss."""
        params = {
            **_DEFAULT_PARAMS,
            "n_estimators": trial.suggest_int("n_estimators", 300, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "early_stopping_rounds": _EARLY_STOP_ROUNDS,
        }
        # Initialize and train model
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # Record iteration
        trial.set_user_attr("best_iteration", model.best_iteration)
        return log_loss(y_val, model.predict_proba(X_val))

    # Initialize optimization study and run trials
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=_OPTUNA_N_TRIALS, show_progress_bar=False)

    # Extract best parameters
    best_n_estimators = study.best_trial.user_attrs["best_iteration"] + 1
    best_params = {k: v for k, v in study.best_trial.params.items() if k != "n_estimators"}

    print(
        f"Optuna best val log-loss: {study.best_value:.4f} | "
        f"n_estimators={best_n_estimators} | params={best_params}"
    )
    return best_params, best_n_estimators


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> XGBClassifier:
    """
    Train an XGBoost multi-class classifier with balanced class weights.
    
    Merges the provided hyperparameters with the default XGBoost settings, computes balanced
    sample weights, and trains the model
    
    Args:
        X_train: Training features.
        y_train: Training labels/outcomes.
        params: A dictionary of hyperparameters to override defaults.
        
    Returns:
        XGBClassifier: A fitted model ready for prediction.
    """
    # Combine default params with optuna search params
    merged = {**_DEFAULT_PARAMS, **params}
    w = compute_sample_weight("balanced", y_train)
    
    # Initialize and train model
    model = XGBClassifier(**merged)
    model.fit(X_train, y_train, sample_weight=w)
    return model


def evaluate(model: XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a trained XGBoost classifier on a held-out test set using multiple metrics.
    
    Computes four complementary metrics:
    - Log-Loss: Confidence penalty for errors
    - Accuracy: Hard-label classification rate
    - Brier Score: Average probability calibration
    - 10-Bin Calibration Curve: Measures how well predicted probabilities match empirical frequencies
    
    Args:
        model: A trained XGBClassifier.
        X_test: Test features.
        y_test: True labels/outcomes.
        
    Returns:
        dict: A metric dictionary with keys: ["logloss", "accuracy", "brier", "calibration"].
    """
    # Compute probabilities and predictions
    proba = model.predict_proba(X_test)
    preds = np.argmax(proba, axis=1)

    # Compute log loss and accuracy
    logloss_val = log_loss(y_test, proba)
    accuracy_val = accuracy_score(y_test, preds)
    
    # Compute Brier score
    brier_val = float(
        np.mean([brier_score_loss((y_test == k).astype(int), proba[:, k]) for k in range(3)])
    )
    
    # Compute and format calibration curve
    frac_pos, mean_pred = calibration_curve(
        y_true=(y_test == 2).astype(int),
        y_prob=proba[:, 2],
        n_bins=10,
        strategy="uniform"
    )
    calibration_pts = list(zip(mean_pred.tolist(), frac_pos.tolist()))

    return {
        "logloss": round(logloss_val, 4),
        "accuracy": round(accuracy_val, 4),
        "brier": round(brier_val, 4),
        "calibration": calibration_pts,
    }


def predict_proba(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """
    Generate calibrated class probability predictions from a trained model.
    
    A validation wrapper around XGBoost's `predict_proba()` method. Enforces that input
    features are 2D before delegating to the model.
    
    Args:
        model: A trained XGBClassifier.
        X: Feature matrix with shape (n_samples, n_features).
        
    Returns:
        np.ndarray: Probability matrix with shape (n_samples, 3)
    """
    assert X.ndim == 2, f"Expected 2-D array, got shape {X.shape}"
    return model.predict_proba(X)


def save_model(model: XGBClassifier, path: Path | str=MODEL_PATH) -> None:
    """
    Save a XGBClassifier model to the provided path.
    
    Args:
        model: A trained XGBClassifier.
        path: The path to save the model to.
        
    Returns:
        None:
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: Path | str=MODEL_PATH) -> XGBClassifier:
    """
    Load a saved XGBClassifier from the provided path.
    
    Args:
        path: The path to the saved model.
        
    Returns:
        XGBClassifier: A XGBClassifier model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path} — run: python src/model.py --retrain"
        )
    return joblib.load(path)


def run_training_pipeline() -> dict:
    """
    End-to-end training pipeline for the WC2026 match predictor.
    
    Orchestrates a complete machine-learning workflow:
    1. Loads or rebuilds the feature matrix from raw match data and rankings
    2. Splits data temporally to prevent data leakage
    3. Runs a 30-trial Bayesian hyperparameter search on train/val to minimize validation log-loss
    4. Retrains on the combined train and val sets with locked hyperparameters
    5. Evaluates the final model on the held-out test set
    6. Saves the fitted model to `models/model.pkl` for serving.
    
    Args:
        None:
        
    Returns:
        dict: A metrics dictionary with keys ["logloss", "accuracy", "brier", "calibration"].
    """
    # Step 1 — load features
    df = _load_or_rebuild_features()

    # Step 2 — temporal split
    X_train, y_train, X_val, y_val, X_test, y_test = _temporal_split(df)

    # Step 3 — hyperparameter search
    print(f"Running Optuna search ({_OPTUNA_N_TRIALS} trials)...")
    best_params, best_n = _run_optuna_search(X_train, y_train, X_val, y_val)

    # Step 4 — refit on train and val splits combined
    X_tv = np.concatenate([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)
    final_params = {**best_params, "n_estimators": best_n}
    print(f"Refitting on train+val ({len(X_tv):,} samples) with n_estimators={best_n}...")
    model = train(X_tv, y_tv, final_params)

    # Step 5 — evaluate on held-out test set
    metrics = evaluate(model, X_test, y_test)
    print(f"\n--- Evaluation on test set (year >= {_TEST_FROM_YEAR}) ---")
    print(f"  Log-loss : {metrics['logloss']:.4f}  (target < 1.0)  {'PASS' if metrics['logloss'] < 1.0 else 'FAIL'}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}  (target > 0.52) {'PASS' if metrics['accuracy'] > 0.52 else 'FAIL'}")
    print(f"  Brier    : {metrics['brier']:.4f}  (target < 0.22)  {'PASS' if metrics['brier'] < 0.22 else 'FAIL'}")

    # Step 6 — save model
    save_model(model, MODEL_PATH)
    return metrics


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # CLI interface
    parser = argparse.ArgumentParser(description="Train WC2026 match predictor.")
    parser.add_argument("--retrain", action="store_true", help="Run full training pipeline")
    args = parser.parse_args()

    # Execute training pipeline or show help
    if args.retrain:
        run_training_pipeline()
    else:
        parser.print_help()
