import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

script_dir = Path(__file__).parent

steps_map = {
    "tokenizer_init": "Tokenizer Init",
    "model_loading": "Model Loading",
    "torch.compile": "torch.compile",
    "graph_capturing": "Graph Capturing",
    "kv_cache_profiling": "KV Cache Profiling",
    "actual_total_time": "Total Time",
} 

# Configuration
base_dir = script_dir / "iterations"
num_iterations = 5
models = ["llama2-13b", "yi-6b", "llama2-7b", "falcon-7b"]
methods = ["safetensors", "tensorizer", "runai_streamer"]
metric = "model_loading"

# Collect data
results = {model: {method: [] for method in methods} for model in models}

for i in range(1, num_iterations + 1):
    path = os.path.join(base_dir, str(i), "comparison_results.json")
    with open(path, "r") as f:
        data = json.load(f)

    labels = data["labels"]
    values = data["data"][metric]

    for label, value in zip(labels, values):
        for model in models:
            if model in label:
                for method in methods:
                    if method in label:
                        results[model][method].append(value)

# Normalize each model's results relative to Safetensors
norm_results = {model: {method: [] for method in methods} for model in models}
for model in models:
    safetensors_values = np.array(results[model]["safetensors"])
    for method in methods:
        vals = np.array(results[model][method])
        norm_results[model][method] = vals # / safetensors_values  # element-wise normalization

# Compute mean and standard error for normalized values
means = []
stderrs = []
for model in models:
    for method in methods:
        vals = np.array(norm_results[model][method])
        means.append(np.mean(vals))
        stderrs.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))

# Plot
x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
for i, method in enumerate(methods):
    method_means = means[i::3]
    method_stderrs = stderrs[i::3]
    bars = ax.bar(x + (i - 1)*width, method_means, width, yerr=method_stderrs,
           label=method.replace("_", " ").title(), capsize=5)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height / 2,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=11, color="white")
    

ax.set_ylabel(f"Model Loading Time (s)", fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels([m.title() for m in models], fontsize=15)
ax.legend(fontsize=13)

plt.tight_layout()
plt.savefig(
    script_dir / "figure14.pdf",
    format="pdf",
    bbox_inches="tight",
    transparent=True
)
plt.show()
