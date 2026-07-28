import argparse
import logging
from pathlib import Path

from src.experiments.ef_grid import FAMILY_EXPERIMENT_SPECS, run_grid_experiment


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3x3 EF experiments for one exponential family.")
    parser.add_argument(
        "family",
        choices=tuple(name for name in FAMILY_EXPERIMENT_SPECS if name != "mixed") + ("all",),
        help="Family to run, or all pure-family experiments.",
    )
    parser.add_argument("--train-size", type=int, default=10_000_000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--lr-size", type=int, default=500 * 250)
    parser.add_argument("--validation-size", type=int, default=20_000)
    parser.add_argument("--many-experiments", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def run_family(name, args):
    spec = FAMILY_EXPERIMENT_SPECS[name]
    run_grid_experiment(
        spec,
        output_dir=Path("outputs") / spec.default_output_dir,
        train_size=args.train_size,
        batch=args.batch,
        lr_size=args.lr_size,
        validation_size=args.validation_size,
        many_experiments=args.many_experiments,
        progress_bar=not args.no_progress,
    )


def main():
    args = parse_args()
    family_names = (
        [name for name in FAMILY_EXPERIMENT_SPECS if name != "mixed"]
        if args.family == "all"
        else [args.family]
    )
    for name in family_names:
        run_family(name, args)


if __name__ == "__main__":
    main()
