from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler

from src.utils.io import save_joblib, load_joblib


@dataclass
class PreprocessorArtifacts:
    transformer_path: str
    scaler_path: str


class Preprocessor:
    """Handle imputation, transformation and scaling consistent with notebook logic."""

    def __init__(
        self,
        skewed_cols: List[str],
        transformer: PowerTransformer | None = None,
        scaler: RobustScaler | None = None,
    ) -> None:
        self.skewed_cols = list(skewed_cols)
        self.transformer = transformer or PowerTransformer(method="yeo-johnson")
        self.scaler = scaler or RobustScaler()

    @staticmethod
    def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        # Drop negative subscription_age rows
        if "subscription_age" in df_out.columns:
            df_out = df_out[df_out["subscription_age"] >= 0]

        # Imputation rules from notebook
        if "reamining_contract" in df_out.columns:
            df_out["reamining_contract"] = df_out["reamining_contract"].fillna(0.0)
        if "download_avg" in df_out.columns:
            df_out["download_avg"] = df_out["download_avg"].fillna(df_out["download_avg"].mean())
        if "upload_avg" in df_out.columns:
            df_out["upload_avg"] = df_out["upload_avg"].fillna(df_out["upload_avg"].mean())
        return df_out

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        df_clean = self.basic_clean(df)
        self.transformer.fit(df_clean[self.skewed_cols])
        transformed = self.transformer.transform(df_clean[self.skewed_cols])
        self.scaler.fit(transformed)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = self.basic_clean(df)
        df_out = df_clean.copy()
        transformed = self.transformer.transform(df_out[self.skewed_cols])
        df_out[self.skewed_cols] = self.scaler.transform(transformed)
        return df_out

    def save(self, artifacts: PreprocessorArtifacts) -> None:
        save_joblib(self.transformer, artifacts.transformer_path)
        save_joblib(self.scaler, artifacts.scaler_path)

    @classmethod
    def load(cls, artifacts: PreprocessorArtifacts, skewed_cols: List[str]) -> "Preprocessor":
        transformer = load_joblib(artifacts.transformer_path)
        scaler = load_joblib(artifacts.scaler_path)
        return cls(skewed_cols=skewed_cols, transformer=transformer, scaler=scaler)


