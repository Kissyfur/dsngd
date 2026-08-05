from dataclasses import dataclass

from src.families import (
    CategoricalCoordinate,
    ExponentialMeanCoordinate,
    GaussianKnownVarianceCoordinate,
    MultivariateGaussianCoordinate,
    PoissonCoordinate,
)


PURE_FAMILY_KEYS = ("categorical", "gaussian", "poisson", "exponential", "multivariate_gaussian")


@dataclass(frozen=True)
class EFExperimentSpec:
    key: str
    title: str
    output_name: str
    default_output_dir: str
    complexity_scenarios: tuple


def categorical(values):
    return lambda: CategoricalCoordinate(values)


def gaussian(variance=1.0):
    return lambda: GaussianKnownVarianceCoordinate(variance=variance)


def multivariate_gaussian(event_dim):
    return lambda: MultivariateGaussianCoordinate(event_dim)


def poisson():
    return PoissonCoordinate


def exponential():
    return ExponentialMeanCoordinate


def paper_discrete_layouts():
    return (
        ("M1", 10, (10, 5)),
        ("M2", 20, (10, 5, 10, 5)),
        ("M3", 30, (10, 5, 10, 5, 10, 5)),
    )


def comparable_scalar_layouts():
    return tuple(
        (name, many_classes, sum(value - 1 for value in feature_values))
        for name, many_classes, feature_values in paper_discrete_layouts()
    )


def comparable_multivariate_layouts():
    return tuple(
        (
            name,
            many_classes,
            closest_multivariate_event_dim(sum(value - 1 for value in feature_values)),
        )
        for name, many_classes, feature_values in paper_discrete_layouts()
    )


def closest_multivariate_event_dim(target_feature_dim):
    event_dim = 1
    while event_dim + event_dim * event_dim < target_feature_dim:
        event_dim += 1
    lower = event_dim - 1
    if lower > 0:
        lower_dim = lower + lower * lower
        upper_dim = event_dim + event_dim * event_dim
        if abs(lower_dim - target_feature_dim) <= abs(upper_dim - target_feature_dim):
            return lower
    return event_dim


def repeated_family_scenarios(factory):
    return tuple(
        (name, many_classes, tuple(factory for _ in range(feature_count)))
        for name, many_classes, feature_count in comparable_scalar_layouts()
    )


def categorical_scenarios():
    return tuple(
        (name, many_classes, tuple(categorical(value) for value in feature_values))
        for name, many_classes, feature_values in paper_discrete_layouts()
    )


def multivariate_gaussian_scenarios():
    return tuple(
        (name, many_classes, (multivariate_gaussian(event_dim),))
        for name, many_classes, event_dim in comparable_multivariate_layouts()
    )


def mixed_repeated_block(repetitions):
    category_values = (10, 5, 10)
    factories = []
    for repetition in range(repetitions):
        factories.extend(
            (
                categorical(category_values[repetition % len(category_values)]),
                gaussian(),
                poisson(),
                exponential(),
            )
        )
    return tuple(factories)


def mixed_repeated_scenarios():
    return (
        ("M1", 10, mixed_repeated_block(1)),
        ("M2", 20, mixed_repeated_block(2)),
        ("M3", 30, mixed_repeated_block(3)),
    )


MIXED_COMPLEXITY_SCENARIOS = (
    ("M1", 4, (categorical(4), gaussian(), poisson())),
    (
        "M2",
        8,
        (
            categorical(5),
            categorical(4),
            gaussian(),
            gaussian(),
            poisson(),
            exponential(),
        ),
    ),
    (
        "M3",
        12,
        (
            categorical(6),
            categorical(5),
            categorical(4),
            gaussian(),
            gaussian(),
            gaussian(),
            poisson(),
            poisson(),
            exponential(),
            exponential(),
        ),
    ),
)


FAMILY_EXPERIMENT_SPECS = {
    "categorical": EFExperimentSpec(
        key="categorical",
        title="Categorical EF",
        output_name="categorical_ef",
        default_output_dir="ef_categorical_experiment",
        complexity_scenarios=categorical_scenarios(),
    ),
    "gaussian": EFExperimentSpec(
        key="gaussian",
        title="Gaussian EF",
        output_name="gaussian_ef",
        default_output_dir="ef_gaussian_experiment",
        complexity_scenarios=repeated_family_scenarios(gaussian()),
    ),
    "poisson": EFExperimentSpec(
        key="poisson",
        title="Poisson EF",
        output_name="poisson_ef",
        default_output_dir="ef_poisson_experiment",
        complexity_scenarios=repeated_family_scenarios(poisson()),
    ),
    "exponential": EFExperimentSpec(
        key="exponential",
        title="Exponential EF",
        output_name="exponential_ef",
        default_output_dir="ef_exponential_experiment",
        complexity_scenarios=repeated_family_scenarios(exponential()),
    ),
    "multivariate_gaussian": EFExperimentSpec(
        key="multivariate_gaussian",
        title="Multivariate Gaussian EF",
        output_name="multivariate_gaussian_ef",
        default_output_dir="ef_multivariate_gaussian_experiment",
        complexity_scenarios=multivariate_gaussian_scenarios(),
    ),
    "mixed_repeated": EFExperimentSpec(
        key="mixed_repeated",
        title="Repeated Mixed EF",
        output_name="mixed_repeated_ef",
        default_output_dir="ef_mixed_repeated_experiment",
        complexity_scenarios=mixed_repeated_scenarios(),
    ),
    "mixed": EFExperimentSpec(
        key="mixed",
        title="Mixed EF",
        output_name="mixed_ef",
        default_output_dir="ef_mixed_experiment",
        complexity_scenarios=MIXED_COMPLEXITY_SCENARIOS,
    ),
}
