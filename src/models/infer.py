from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.features.preprocess import Preprocessor, PreprocessorArtifacts
from src.utils.io import load_joblib


@dataclass
class InferenceArtifacts:
    model_path: str
    transformer_path: str
    scaler_path: str
    skewed_cols: List[str]
    target: str


class InferenceService:
    def __init__(self, artifacts: InferenceArtifacts) -> None:
        self.model = load_joblib(artifacts.model_path)
        preproc_art = PreprocessorArtifacts(
            transformer_path=artifacts.transformer_path,
            scaler_path=artifacts.scaler_path,
        )
        self.preprocessor = Preprocessor.load(preproc_art, skewed_cols=artifacts.skewed_cols)
        self.target = artifacts.target

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # Remove ID column if it exists (same logic as training)
        df_clean = df.copy()
        if 'id' in df_clean.columns:
            df_clean = df_clean.drop(columns=['id'])
        
        processed = self.preprocessor.transform(df_clean)
        X = processed.drop(columns=[self.target]) if self.target in processed.columns else processed
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(df)
        return (probs >= threshold).astype(int)


