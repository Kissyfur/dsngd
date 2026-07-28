import logging
from pathlib import Path

from src.experiments.ef_grid import FAMILY_EXPERIMENT_SPECS, run_grid_experiment


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    run_grid_experiment(
        FAMILY_EXPERIMENT_SPECS["mixed"],
        output_dir=Path("outputs") / "ef_mixed_experiment",
    )


if __name__ == "__main__":
    main()
