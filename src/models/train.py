from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import optuna

from src.features.preprocess import Preprocessor, PreprocessorArtifacts
from src.utils.io import read_csv, ensure_dir, save_joblib


@dataclass
class TrainResult:
    model_path: str
    cv_scores: np.ndarray
    cv_mean: float


def prepare_dataframe(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    df_out = df.copy()
    if id_column in df_out.columns:
        df_out = df_out.drop(columns=[id_column])
    return df_out


def build_model(params: Dict) -> XGBClassifier:
    return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, **params)


def optimize_hyperparams(X: pd.DataFrame, y: pd.Series, base_params: Dict, cv_folds: int, n_trials: int) -> Dict:
    def objective(trial: optuna.Trial) -> float:
        params = base_params.copy()
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 50, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            }
        )
        model = build_model(params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    base_params.update(best_params)
    return base_params


def train(cfg) -> TrainResult:
    logger.info("Loading data...")
    train_csv_path = Path(cfg.paths.train_csv)
    test_csv_path = Path(cfg.paths.test_csv)
    raw_path = Path(cfg.paths.raw_data)

    data = read_csv(train_csv_path) if train_csv_path.exists() else read_csv(raw_path)

    if not train_csv_path.exists() or not test_csv_path.exists():
        logger.info("Train/test CSV not found. Creating split from raw data...")
        train_df, test_df = train_test_split(
            data, test_size=cfg.training.test_size, stratify=data[cfg.features.target], random_state=cfg.training.random_state
        )
        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)
        data = train_df

    # Prepare and preprocess
    df = prepare_dataframe(data, cfg.features.id_column)
    preprocessor = Preprocessor(skewed_cols=list(cfg.features.skewed_cols))
    preprocessor.fit(df)
    df_pre = preprocessor.transform(df)

    # Train model
    X = df_pre.drop(columns=[cfg.features.target])
    y = df_pre[cfg.features.target]

    params = dict(cfg.training.xgboost_params)
    if cfg.training.use_optuna:
        logger.info("Optimizing hyperparameters with Optuna...")
        params = optimize_hyperparams(X, y, params, cfg.training.cv_folds, cfg.training.optuna_trials)

    model = build_model(params)

    # MLflow
    # Ensure runs go to the configured tracking server (defaults to http://localhost:5000)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=False)
    with mlflow.start_run(run_name="train"):
        cv = StratifiedKFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=cfg.training.random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        mlflow.log_metric("cv_mean_accuracy", float(np.mean(scores)))
        logger.info(f"CV accuracy: {scores} | Mean: {np.mean(scores):.4f}")

        model.fit(X, y)

        # Save artifacts
        artifacts = PreprocessorArtifacts(
            transformer_path=cfg.paths.transformer_path,
            scaler_path=cfg.paths.scaler_path,
        )
        preprocessor.save(artifacts)
        model_path = Path(cfg.paths.model_path)
        ensure_dir(model_path.parent)
        save_joblib(model, model_path)
        mlflow.log_artifact(str(model_path))

    return TrainResult(model_path=cfg.paths.model_path, cv_scores=scores, cv_mean=float(np.mean(scores)))


