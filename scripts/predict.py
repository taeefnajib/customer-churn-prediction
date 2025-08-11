from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import Optional

import pandas as pd
import typer
from hydra import compose, initialize

from src.models.infer import InferenceService, InferenceArtifacts
from src.utils.logging import configure_logging


app = typer.Typer()


def load_cfg():
    with initialize(version_base=None, config_path="../src/config"):
        cfg = compose(config_name="config.yaml")
        return cfg


@app.command()
def predict(input_csv: Optional[Path] = None, input_json: Optional[str] = None, threshold: float = 0.5):
    """Predict churn for a CSV file or a single JSON record."""
    configure_logging()
    cfg = load_cfg()
    artifacts = InferenceArtifacts(
        model_path=cfg.paths.model_path,
        transformer_path=cfg.paths.transformer_path,
        scaler_path=cfg.paths.scaler_path,
        skewed_cols=list(cfg.features.skewed_cols),
        target=cfg.features.target,
    )
    service = InferenceService(artifacts)

    if input_csv:
        df = pd.read_csv(input_csv)
        probs = service.predict_proba(df)
        preds = (probs >= threshold).astype(int)
        out = df.copy()
        out["predicted_churn"] = preds
        out["churn_probability"] = probs
        out.to_csv("predictions.csv", index=False)
        typer.echo("Saved predictions.csv")
    elif input_json:
        record = json.loads(input_json)
        df = pd.DataFrame([record])
        prob = float(service.predict_proba(df)[0])
        pred = int(prob >= threshold)
        typer.echo(json.dumps({"predicted_churn": pred, "churn_probability": prob}))
    else:
        typer.echo("Provide --input-csv PATH or --input-json '{...}'")


if __name__ == "__main__":
    app()


