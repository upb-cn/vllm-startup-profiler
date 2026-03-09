#!/bin/bash

EXP_MACHINE=${1:-node7}
EXP_DIR=${2:-model_size}

DIR_PATH="experiments/$EXP_MACHINE/examples/$EXP_DIR"

# Loop through all iterations in $DIR_PATH/iterations
for i in $(ls -d $DIR_PATH/iterations/*/ | xargs -n 1 basename); do
    python3 compare_logs.py $DIR_PATH/iterations/$i/output_model_* $DIR_PATH/iterations/$i/comparison_results.json;
done

# Check if uncached exists
if [ ! -d "$DIR_PATH/uncached" ]; then
    echo "No uncached directory found at $DIR_PATH/uncached. Skipping uncached comparison."
else
    python3 compare_logs.py $DIR_PATH/uncached/output_model_* $DIR_PATH/uncached/comparison_results.json;
fi

python3 avg_results.py $DIR_PATH/iterations