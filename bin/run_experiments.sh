#!/usr/bin/env bash
# Runs the experiments of the paper, at most JOBS at a time, writing results and logs to outputs/. The second argument
# runs only the synthetic or only the real data experiments. Each line of a log reports the elapsed time and the
# expected remaining time of its experiment.
# Usage: [JOBS=6] [EXPERIMENTS=1] bin/run_experiments.sh [python] [all|synthetic|real]
PYTHON=${1:-python}
PART=${2:-all}
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
    echo "$(date +%H:%M) started $log"
    "$@" > "$log" 2>&1 &
}

if [ "$PART" != real ]; then
    for features in discrete gaussian gaussian-mv poisson exponential gamma beta multivariate-gaussian dirichlet mixed; do
        run outputs/synthetic/$features.log $PYTHON bin/experiment.py --features $features --experiments $EXPERIMENTS \
            --quiet --output outputs/synthetic/$features
    done
fi
if [ "$PART" != synthetic ]; then
    for design in covertype:natural covertype:gaussian mnist:levels mnist:bernoulli mnist:gaussian; do
        dataset=${design%%:*}
        features=${design##*:}
        run outputs/real/${dataset}_$features.log $PYTHON bin/experiment_real.py --dataset $dataset \
            --features $features --quiet --output outputs/real/${dataset}_$features
    done
fi
wait
echo "$(date +%H:%M) all finished"
