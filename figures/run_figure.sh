export VLLM_LOGGING_CONFIG_PATH=$(realpath ../vllm_logging_config.json)

FIGURE=${1:-1}
CURRENT_HARDWARE_INDEX=${2:-0}

FIGURE_DIR="figure-$FIGURE"
DIR_PATH="figures/$FIGURE_DIR"

# Used for figure 10 & 11 & 13

if [ "$FIGURE" = "1" ]; then
    # Need to create a virtual environment for every vllm version
    # and then run the test

    cd $FIGURE_DIR

    for version in "0.3.1" "0.4.0" "0.5.0" "0.6.0" "0.7.0" "0.8.0" "0.9.0" "0.10.0" "0.11.0"; do
        python3 -m venv .venv-$version
        source .venv-$version/bin/activate

        pip install vllm==$version

        # Install correct transformers version
        if [ "$version" = "0.3.1" ]; then
            pip install transformers==4.37.0
        elif [ "$version" = "0.4.0" ]; then
            pip install transformers==4.39.1
            pip install numpy==1.26.4
        elif [ "$version" = "0.5.0" ]; then
            pip install transformers==4.40.0
        elif [ "$version" = "0.6.0" ]; then
            pip install transformers==4.43.2
        elif [ "$version" = "0.7.0" ]; then
            pip install transformers==4.45.2
        elif [ "$version" = "0.8.0" ]; then
            pip install transformers==4.48.2
        elif [ "$version" = "0.9.0" ]; then
            pip install transformers==4.51.1
        elif [ "$version" = "0.10.0" ]; then
            pip install transformers==4.53.2
        elif [ "$version" = "0.11.0" ]; then
            pip install transformers==4.55.2
        fi

        export VLLM_LOGGING_LEVEL=DEBUG
        export VLLM_USE_V1=1

        # First one is warmup
        python3 ../../engine.py --model ../../models/opt-6.7b | tee iterations/output_${version}.txt
        python3 ../../engine.py --model ../../models/opt-6.7b | tee iterations/output_${version}.txt
    done

    
    # Plot figure
    python3 plot.py

elif [ "$FIGURE" = "2" ] || [ "$FIGURE" = "rest" ] || [ "$FIGURE" = "7" ] || [ "$FIGURE" = "9" ] || [ "$FIGURE" = "10" ] || [ "$FIGURE" = "11" ] || [ "$FIGURE" = "13" ] || [ "$FIGURE" = "14" ] || [ "$FIGURE" = "15" ]; then

    run_test() {
        # Clear RAM to force weights to be retrieved from SSD
        if [ "$FIGURE" = "13" ] && [ "$CURRENT_HARDWARE_INDEX" = "1" ]; then
            sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
        fi

        python3 run_test.py $DIR_PATH/env.json
    }

    mv_outputs() {
        mv $DIR_PATH/output* "$1"
        mv $DIR_PATH/comparison_results.json "$1"
    }

    cd ..

    # Warmup run
    run_test
    mkdir -p "$DIR_PATH/uncached"
    mv_outputs "$DIR_PATH/uncached"

    for i in $(seq 1 5); do
        mkdir -p "$DIR_PATH/iterations/$i"
        run_test
        mv_outputs "$DIR_PATH/iterations/$i";
    done

    python3 avg_results.py $DIR_PATH/iterations

    # Fix comparisons in case of new uncompleted runs
    for i in $(ls -d $DIR_PATH/iterations/*/ | xargs -n 1 basename); do
        python3 compare_logs.py $DIR_PATH/iterations/$i/output_model_* $DIR_PATH/iterations/$i/comparison_results.json;
    done

    # Check if uncached exists
    if [ -d "$DIR_PATH/uncached" ]; then
        python3 compare_logs.py $DIR_PATH/uncached/output_model_* $DIR_PATH/uncached/comparison_results.json;
    fi

    if [ "$FIGURE" = "10" ] || [ "$FIGURE" = "11" ] || [ "$FIGURE" = "13" ]; then
        if [ "$FIGURE" = "10" ]; then
            DIR_NAME=gpu
        elif [ "$FIGURE" = "11" ]; then
            DIR_NAME=cpu
        else
            DIR_NAME=case
        fi
        mv $DIR_PATH/iterations $DIR_PATH/uncached $DIR_PATH/avg_comparison_results.json $DIR_PATH/${DIR_NAME}${CURRENT_HARDWARE_INDEX}
    fi

    if [ "$FIGURE" = "rest" ]; then
        mkdir -p $DIR_PATH/predictor_info
        # Create Figure 17
        python3 predictor/train_predictor.py \
            --avg_comparison_results $DIR_PATH/avg_comparison_results.json \
            --models_config models_config.json \
            --models_output_dir $DIR_PATH/predictor_info/models \
            --ignore_models "falcon-11b,gemma-7b,mistral-7b,llama3-3b,qwen-7b,qwen-14.3b,gpt-oss-20b,deepseek-v2-lite-16b,granite4.0-h-32b"

        python3 predictor/create_test_data.py \
            --avg_comparison_results $DIR_PATH/avg_comparison_results.json \
            --models_config models_config.json \
            --output_path $DIR_PATH/predictor_info/ \
            --validation_models "falcon-11b,gemma-7b,mistral-7b,llama3-3b,qwen-7b"
    fi

    cd -

    # Plot figure
    python3 $FIGURE_DIR/plot.py

elif [ "$FIGURE" = "12" ]; then
    cd figure-12
    export VLLM_USE_V1=1
    export VLLM_LOGGING_LEVEL=DEBUG

    # Warm up, then test
    python3 engine.py --model ../../models/qwen-4b | tee qwen-4b.txt
    python3 engine.py --model ../../models/qwen-4b | tee qwen-4b.txt

    python3 plot.py
else
    echo "Invalid figure number: $FIGURE. Please provide one of the following: '1', '2', '9', '10', '11', '12', '13', '14', '15', '17', 'rest'."
    exit 1
fi