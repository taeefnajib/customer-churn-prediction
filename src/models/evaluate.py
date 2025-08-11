from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import mlflow

from src.features.preprocess import Preprocessor, PreprocessorArtifacts
from src.utils.io import read_csv, load_joblib


def evaluate(cfg) -> Dict[str, float]:
    logger.info("Loading test data and artifacts...")
    test_df = read_csv(cfg.paths.test_csv)

    # Remove ID column if it exists (same logic as training)
    if cfg.features.id_column in test_df.columns:
        test_df = test_df.drop(columns=[cfg.features.id_column])
        logger.info(f"Removed ID column: {cfg.features.id_column}")

    artifacts = PreprocessorArtifacts(
        transformer_path=cfg.paths.transformer_path,
        scaler_path=cfg.paths.scaler_path,
    )
    preprocessor = Preprocessor.load(artifacts, skewed_cols=list(cfg.features.skewed_cols))
    model = load_joblib(cfg.paths.model_path)

    X_test = preprocessor.transform(test_df).drop(columns=[cfg.features.target])
    y_test = test_df[cfg.features.target]

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= cfg.ui.default_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
    }

    logger.info(f"Metrics: {metrics}")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name="evaluate"):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    return metrics


def compute_youden_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    j_max_idx = int(np.argmax(j_scores))
    return float(thresholds[j_max_idx])


