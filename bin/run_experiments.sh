#!/usr/bin/env bash
# Runs the experiments of the paper, at most JOBS at a time, writing results and logs to outputs/. The second argument
# runs only the synthetic or only the real data experiments, and further arguments only the given experiments, such as
# gaussian or mnist:gaussian. MANIFOLDS and ENTROPIES run only some manifolds and entropies of the synthetic
# experiments, whose results replace theirs in outputs/ while those of the others are kept. Each line of a log reports
# the elapsed time and the expected remaining time of its experiment.
# Usage: [JOBS=6] [EXPERIMENTS=1] [MANIFOLDS="1 2 3"] [ENTROPIES="high medium low"] bin/run_experiments.sh [python]
#        [all|synthetic|real] [experiment ...]
PYTHON=${1:-python}
PART=${2:-all}
ONLY="${*:3}"
JOBS=${JOBS:-6}
EXPERIMENTS=${EXPERIMENTS:-1}
# The slowest synthetic experiments first, so that the parallel jobs end at about the same time
SYNTHETIC="dirichlet beta gamma multivariate-gaussian mixed gaussian-mv discrete gaussian poisson exponential"
REAL="covertype:natural covertype:gaussian mnist:levels mnist:bernoulli mnist:gaussian"
export PYTHONPATH=. MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2
mkdir -p outputs/synthetic outputs/real

for name in $ONLY; do
    if [[ " $SYNTHETIC $REAL " != *" $name "* ]]; then
        echo "Unknown experiment $name, the experiments are $SYNTHETIC $REAL"
        exit 1
    fi
done

selected() {
    # Whether the experiment is among those given, if any
    [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]
}

run() {
    local log=$1
    shift
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        wait -n
    done
    echo "$(date +%H:%M) started $log"
    "$@" > "$log" 2>&1 &
}

# Some manifolds and entropies only, with a log of their own
CELLS=()
SUFFIX=
if [ -n "$MANIFOLDS" ]; then
    CELLS+=(--manifolds $MANIFOLDS)
    SUFFIX+=_M${MANIFOLDS// /}
fi
if [ -n "$ENTROPIES" ]; then
    CELLS+=(--entropies $ENTROPIES)
    SUFFIX+=_${ENTROPIES// /_}
fi

if [ "$PART" != real ]; then
    for features in $SYNTHETIC; do
        selected $features || continue
        run outputs/synthetic/$features$SUFFIX.log $PYTHON bin/experiment.py --features $features \
            --experiments $EXPERIMENTS "${CELLS[@]}" --quiet --output outputs/synthetic/$features
    done
fi
if [ "$PART" != synthetic ]; then
    for design in $REAL; do
        selected $design || continue
        dataset=${design%%:*}
        features=${design##*:}
        run outputs/real/${dataset}_$features.log $PYTHON bin/experiment_real.py --dataset $dataset \
            --features $features --quiet --output outputs/real/${dataset}_$features
    done
fi
wait
echo "$(date +%H:%M) all finished"
