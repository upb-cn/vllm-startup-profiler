import os
import json
import numpy as np
import sys

# Path to the iterations directory
if len(sys.argv) < 2:
    print("Usage: python avg_results.py <iterations_dir_path>")
    sys.exit(1)
    
iterations_dir_path = sys.argv[1]

# Output file should be sibiling to iterations_dir_path
parent_dir = os.path.dirname(iterations_dir_path.rstrip(os.sep))
output_file = os.path.join(parent_dir, "avg_comparison_results.json")

# Initialize containers
labels = None
data_accumulator = {}  # key -> list of lists (each sublist is from one file)

# Gather and process all JSON files
for iteration in sorted(os.listdir(iterations_dir_path)):
    file_path = os.path.join(iterations_dir_path, iteration, "comparison_results.json")
    if not os.path.isfile(file_path):
        continue

    with open(file_path, "r") as f:
        content = json.load(f)

    # Save labels (assumed to be the same across files)
    if labels is None:
        labels = content["labels"]

    for key, values in content["data"].items():
        if key not in data_accumulator:
            data_accumulator[key] = [[] for _ in range(len(values))]

        for i, val in enumerate(values):
            if val is not None:
                
                # This is important to fix actual_total_time later
                # The idea is that total_time should be the actual sum of the avgs
                # Then actual_total_time should be the real total_time + averaged additional time 
                if key == "actual_total_time":
                    val -= content["data"]["total_time"][i]
                
                data_accumulator[key][i].append(val)

# Compute mean
aggregated_mean_data = {}
for key, value_lists in data_accumulator.items():
    mean = []
    variance = []

    for values in value_lists:
        if values:
            mean.append(float(np.mean(values)))
        else:
            mean.append(None)

    aggregated_mean_data[key] = mean

def sum_up_keys(data, i, keys):
    total = 0
    for k in keys:
        val = data[k][i]
        if val is not None:
            total += val
            
    return total
# Fix keys that are usually sum up of other keys
for i, value in enumerate(aggregated_mean_data["total_time"]):
    
    framework_bootstrap = sum_up_keys(aggregated_mean_data, i, ["detect_platform", "llm_imports", "get_model_info", "worker_init"])
    model_loading = sum_up_keys(aggregated_mean_data, i, ["load_weights", "model_init"])
    torch_compile = sum_up_keys(aggregated_mean_data, i, ["dynamo_transform_time", "graph_compile_general_shape"])
    init_engine = torch_compile + sum_up_keys(aggregated_mean_data, i, ["kv_cache_profiling", "graph_compile_cached", "graph_capturing"])
    total_time = framework_bootstrap + model_loading + init_engine + sum_up_keys(aggregated_mean_data, i, ["tokenizer_init"])
    actual_total_time = total_time + sum_up_keys(aggregated_mean_data, i, ["actual_total_time"])
    
    aggregated_mean_data["framework_bootstrap"][i] = framework_bootstrap
    aggregated_mean_data["model_loading"][i] = model_loading
    aggregated_mean_data["torch.compile"][i] = torch_compile
    aggregated_mean_data["init_engine"][i] = init_engine
    aggregated_mean_data["total_time"][i] = total_time
    aggregated_mean_data["actual_total_time"][i] = actual_total_time


# Save the aggregated results
output = {
    "labels": labels,
    "data": aggregated_mean_data,
}

with open(output_file, "w") as f:
    json.dump(output, f, indent=4)

print(f"Saved averaged results to: {output_file}")