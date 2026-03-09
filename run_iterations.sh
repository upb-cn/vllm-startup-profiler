#!/bin/bash

EXP_MACHINE=${1:-node7}
EXP_DIR=${2:-model_size}
IS_FIRST_TIME=${3:-false}
NUM_ITERATIONS=${4:-5}

DIR_PATH="experiments/$EXP_MACHINE/examples/$EXP_DIR"

run_test() {
    python3 run_test.py $DIR_PATH/env.json
}

mv_outputs() {
    mv $DIR_PATH/output* "$1"
    mv $DIR_PATH/comparison_results.json "$1"
}

# is it first time?
if [ "$IS_FIRST_TIME" = true ] ; then
    run_test
    mkdir -p "$DIR_PATH/uncached"
    mv_outputs "$DIR_PATH/uncached"
fi

for i in $(seq 1 $NUM_ITERATIONS); do
    mkdir -p "$DIR_PATH/iterations/$i"
    run_test
    mv_outputs "$DIR_PATH/iterations/$i";
done

python3 avg_results.py $DIR_PATH/iterations