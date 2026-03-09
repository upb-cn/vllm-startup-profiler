# This is a custom script to handle the differences in vLLM versions
# I am only interested in the actual total time, and nothing else

import os
import re
import json
import sys
sys.path.append(os.path.abspath(os.path.join("..", "..", "..", "..")))
from utils import extract_version

avg_comparison_results = {}
results = {}
labels = []

iterations_dir_path = "iterations/"
# Loop over all iterations
for iter_dir in os.listdir(iterations_dir_path):
    iter_path = os.path.join(iterations_dir_path, iter_dir)
    if os.path.isdir(iter_path):
        # Loop over all versions outputs
        for file_name in os.listdir(iter_path):
            if file_name not in labels: labels.append(file_name)
            file_path = os.path.join(iter_path, file_name)
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "initialize_engine took" in line:
                        match = re.search(r"initialize_engine took ([\d.]+) seconds", line)
                        actual_total_time = float(match.group(1))
                        if file_name not in results:
                            results[file_name] = []
                        results[file_name].append(actual_total_time)

labels = sorted(labels, key=lambda label: extract_version(label))
avg_comparison_results["labels"] = labels
avg_comparison_results["data"] = {
    "actual_total_time": [sum(results[label])/len(results[label]) for label in labels]
}
other_keys = ["load_weights", "model_init", "model_loading", "dynamo_transform_time",
              "graph_compile_general_shape", "graph_compile_cached", "torch.compile",
              "kv_cache_profiling", "graph_capturing", "init_engine", "tokenizer_init", "total_time"]
for key in other_keys:
    avg_comparison_results["data"][key] = [0 for _ in labels]

with open("avg_comparison_results.json", "w") as f:
    json.dump(avg_comparison_results, f, indent=4)