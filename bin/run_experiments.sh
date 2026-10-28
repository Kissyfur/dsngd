#!/usr/bin/env bash
# Runs the experiments of the paper, at most JOBS at a time, writing results and logs to outputs/.
# Usage: [JOBS=6] [EXPERIMENTS=1] bin/run_experiments.sh [python]
PYTHON=${1:-python}
JOBS=${JOBS:-6}
EXPERIMENTS=${EXPERIMENTS:-1}
export PYTHONPATH=. MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2
mkdir -p outputs/synthetic outputs/real

run() {
    local log=$1
    shift
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        wait -n
    done
    "$@" > "$log" 2>&1 &
}

for features in discrete gaussian gaussian-mv poisson exponential gamma beta multivariate-gaussian dirichlet mixed; do
    run outputs/synthetic/$features.log $PYTHON bin/experiment.py --features $features --experiments $EXPERIMENTS \
        --quiet --output outputs/synthetic/$features
done
run outputs/real/covertype.log $PYTHON bin/experiment_real.py --dataset covertype --quiet --output outputs/real/covertype
run outputs/real/mnist_gaussian.log $PYTHON bin/experiment_real.py --dataset mnist --quiet \
    --output outputs/real/mnist_gaussian
run outputs/real/mnist_bernoulli.log $PYTHON bin/experiment_real.py --dataset mnist --mnist-features bernoulli --quiet \
    --output outputs/real/mnist_bernoulli
wait
