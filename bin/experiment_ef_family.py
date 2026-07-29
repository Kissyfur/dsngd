import argparse
import logging
from pathlib import Path

from src.experiments.ef_grid import (
    DEFAULT_EVAL_VALIDATION_SIZE,
    DEFAULT_LR_VALIDATION_SIZE,
    run_grid_experiment,
)
from src.experiments.ef_specs import FAMILY_EXPERIMENT_SPECS, PURE_FAMILY_KEYS


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3x3 EF experiments for one exponential family.")
    parser.add_argument(
        "family",
        choices=PURE_FAMILY_KEYS + ("all",),
        help="Family to run, or all pure-family experiments.",
    )
    parser.add_argument("--train-size", type=int, default=10_000_000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--lr-size", type=int, default=500 * 250)
    parser.add_argument(
        "--lr-validation-size",
        type=int,
        default=DEFAULT_LR_VALIDATION_SIZE,
        help="Held-out sample used only to score candidate learning rates.",
    )
    parser.add_argument(
        "--eval-validation-size",
        "--validation-size",
        dest="eval_validation_size",
        type=int,
        default=DEFAULT_EVAL_VALIDATION_SIZE,
        help="Fresh held-out sample used only for final reported curves.",
    )
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
        lr_validation_size=args.lr_validation_size,
        eval_validation_size=args.eval_validation_size,
        many_experiments=args.many_experiments,
        progress_bar=not args.no_progress,
    )


def main():
    args = parse_args()
    family_names = (
        list(PURE_FAMILY_KEYS)
        if args.family == "all"
        else [args.family]
    )
    for name in family_names:
        run_family(name, args)


if __name__ == "__main__":
    main()
