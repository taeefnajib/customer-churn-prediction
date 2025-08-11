from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hydra import compose, initialize
from src.models.evaluate import evaluate
from src.utils.logging import configure_logging


def main() -> None:
    configure_logging()
    with initialize(version_base=None, config_path="../src/config"):
        cfg = compose(config_name="config")
    evaluate(cfg)


if __name__ == "__main__":
    main()


