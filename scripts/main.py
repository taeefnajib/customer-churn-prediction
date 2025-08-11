from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
from hydra import compose, initialize
from omegaconf import OmegaConf

from src.utils.logging import configure_logging, logger
from src.models.train import train
from src.models.evaluate import evaluate


app = typer.Typer(help="Customer Churn - CLI")


def load_cfg():
    with initialize(version_base=None, config_path="../src/config"):
        cfg = compose(config_name="config")
        return cfg


@app.command(name="train")
def train_cmd() -> None:
    """Train model and log to MLflow."""
    configure_logging()
    cfg = load_cfg()
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    train(cfg)


@app.command(name="evaluate")
def evaluate_cmd() -> None:
    """Evaluate model on test set and log metrics to MLflow."""
    configure_logging()
    cfg = load_cfg()
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    evaluate(cfg)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Launch Gradio UI."""
    configure_logging()
    from app.gradio_app import launch_app

    cfg = load_cfg()
    launch_app(cfg, host=host, port=port)


if __name__ == "__main__":
    app()


