import logging
from pathlib import Path

from src.experiments.ef_grid import run_grid_experiment
from src.experiments.ef_specs import FAMILY_EXPERIMENT_SPECS


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    run_grid_experiment(
        FAMILY_EXPERIMENT_SPECS["mixed_repeated"],
        output_dir=Path("outputs") / "ef_mixed_repeated_experiment",
    )


if __name__ == "__main__":
    main()
