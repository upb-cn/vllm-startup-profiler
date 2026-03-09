import json
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import argparse
from pathlib import Path
import pickle
import re
import os
import time
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import extract_model_name

def create_predictor_wrt_key(key, metric, data, models_output_dir):
    """Create predictor for key based on metric"""
    
    # Extract configs and timings
    configs = data["models_config"]
    timings = data["data"]
    labels = data["labels"]
    
    key_values = np.array([configs[label][key] for label in labels]).reshape(-1, 1)
    times = np.array(timings[metric])

    model = LinearRegression()
    model.fit(key_values, times)
    
    joblib.dump(model, f"{models_output_dir}/{metric}.pkl")

def create_predictor_graph_capturing_time(data, models_output_dir):
    """Create predictor for graph capturing given model size (in B) and batch size."""
    
    # Extract configs and timings
    configs = data["models_config"]
    timings = data["data"]
    labels = data["labels"]
    
    size_batch_product = np.array([
        configs[label]["size"] * 67 for label in labels
    ]).reshape(-1, 1)

    times_graph = np.array(timings["graph_capturing"])

    model_graph = LinearRegression()
    model_graph.fit(size_batch_product, times_graph)
    
    joblib.dump(model_graph, f"{models_output_dir}/graph_capturing.pkl")
    
def create_constant_predictor(key, data, models_output_dir):
    """Create constant predictor"""
    # Currently, this is just for model_init
    
    key_times = data["data"][key]
    avg = sum(key_times) / len(key_times)
    
    with open(f"{models_output_dir}/constant_{key}.pkl", "wb") as f:
        pickle.dump(avg, f)
    
def create_predictors(predictor_input, models_output_dir):
    os.makedirs(models_output_dir, exist_ok=True)
    
    key_metric_comb = [
        ("size", "load_weights"),
        ("compiled_graph_sizes", "dynamo_transform_time"),
        ("compiled_graph_sizes", "graph_compile_cached"),
        ("size", "kv_cache_profiling"),
        ("tokenizer_size", "tokenizer_init")
    ]
    for key_metric in key_metric_comb:
        create_predictor_wrt_key(key_metric[0], key_metric[1], predictor_input, models_output_dir)
    create_predictor_graph_capturing_time(predictor_input, models_output_dir)
    create_constant_predictor("model_init", predictor_input, models_output_dir)
    create_constant_predictor("framework_bootstrap", predictor_input, models_output_dir)

def parse_input(avg_comparison_results_path, models_config_path, ignore_models):
    with open(avg_comparison_results_path, "r") as f:
        results = json.load(f)
        
    with open(models_config_path, "r") as f:
        models_config = json.load(f)
    
    ignore_models_list = ignore_models.split(",") if ignore_models else []
     
    # Prepare the labels   
    labels = []
    ignored_indices = set()
    for i,l in enumerate(results["labels"]):
        model_name = extract_model_name(l)
        if model_name in ignore_models_list:
            ignored_indices.add(i)
            continue
            
        labels.append(model_name)

    # Prepare the data
    for key in results["data"].keys():
        results["data"][key] = [
            v for i,v in enumerate(results["data"][key]) if i not in ignored_indices
        ]
    data = results["data"]
    
    output = {
        "labels": labels,
        "data": data,
        "models_config": models_config
    }
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the predictors")
    parser.add_argument("--avg_comparison_results", help="Path to avg_comparison_results.json", type=Path, required=True)
    parser.add_argument("--models_config", help="Path to models_config.json", type=Path, required=True)
    parser.add_argument("--models_output_dir", help="Path to dump output models", type=Path, required=True)
    parser.add_argument("--ignore_models", help="Comma separated list of models to ignore", type=str)
    args = parser.parse_args()
    
    predictor_input = parse_input(args.avg_comparison_results, args.models_config, args.ignore_models)
    start = time.perf_counter()
    create_predictors(predictor_input, args.models_output_dir)
    end = time.perf_counter()
    
    print(f"Predictor models have been created successfully in {((end - start)*1000):.2f} ms!")