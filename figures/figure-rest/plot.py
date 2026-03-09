import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)
from utils import *
from pathlib import Path

script_dir = Path(__file__).parent
iterations_path = script_dir / "iterations"

    
# --- Figure 3 ---
model_names_map = {
    "llama2-7b-hf": "Llama2-7B",
    "llama2-13b-hf": "Llama2-13B",
    "falcon-7b": "Falcon-7B",
    "yi-6b": "Yi-6B",
    "yi-9b": "Yi-9B",
    "qwen-0.5b": "Qwen-0.5B",
    "qwen-1.8b": "Qwen-1.8B",
    "qwen-4b": "Qwen-4B",
    "qwen-14b": "Qwen-14B",
    "llama3-3b": "Llama3-3B"
}
sort_by = "model_size"
metric1 = "tokenizer_init"
metric2 = "tokenizer_size"
xlabel = ""
ylabel = "Tokenizer Initialization Time (s)"
x2label = "Tokenizer Size (MB)"
y2label = "Init Time (s)"
ylim_multiplier = 1.2
filename = script_dir / "figure3"
draw(model_names_map, iterations_path, sort_by, metric1, metric2, xlabel, ylabel, x2label, y2label, ylim_multiplier, filename)

# --- Figure 4 ---
model_names_map = {
    "llama2-7b-hf": "Llama2-7B",
    "llama2-13b-hf": "Llama2-13B",
    "yi-6b": "Yi-6B",
    "yi-9b": "Yi-9B",
    "qwen-1.8b": "Qwen-1.8B",
    "qwen-4b": "Qwen-4B",
    "qwen-14b": "Qwen-14B",
    "qwen-14.3b": "Qwen-MoE-14.3B",
    "llama3-3b": "Llama3-3B",
    "deepseek-v2-lite-16b": "DeepSeek-V2-Lite-16B"
}
sort_by = "model_size"
metric1 = "load_weights"
metric2 = "size"
xlabel = ""
ylabel = "Loading Weights Time (s)"
x2label = "Model Size (B)"
y2label = "Loading Time (s)"
ylim_multiplier = 1.1
filename = script_dir / "figure4"
draw(model_names_map, iterations_path, sort_by, metric1, metric2, xlabel, ylabel, x2label, y2label, ylim_multiplier, filename)

# --- Figure 5 ---
model_names_map = {
    "llama2-7b-hf": "Llama2-7B",
    "llama2-13b-hf": "Llama2-13B",
    "yi-6b": "Yi-6B",
    "qwen-4b": "Qwen-4B",
    "qwen-14b": "Qwen-14B",
    "llama3-3b": "Llama3-3B",
    "deepseek-v2-lite-16b": "DeepSeek-V2-Lite-16B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "granite3.3-8b-instruct": "Granite3.3-8B-Instruct",
    "granite4.0-h-32b": "Granite4.0-h-32B"
}
sort_by = "model_size"
metric1 = "dynamo_transform_time"
metric2 = "compiled_graph_sizes"
xlabel = ""
ylabel = "Dynamo Transformation Time (s)"
x2label = "Size (KB)"
y2label = "Transform Time (s)"
ylim_multiplier = 1.5
filename = script_dir / "figure5"
draw(model_names_map, iterations_path, sort_by, metric1, metric2, xlabel, ylabel, x2label, y2label, ylim_multiplier, filename)


# --- Figure 6 ---
model_names_map = {
    "llama2-7b-hf": "Llama2-7B",
    "llama2-13b-hf": "Llama2-13B",
    "yi-9b": "Yi-9B",
    "qwen-4b": "Qwen-4B",
    "qwen-14b": "Qwen-14B",
    "qwen-14.3b": "Qwen-MoE-14.3B",
    "deepseek-v2-lite-16b": "DeepSeek-V2-Lite-16B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "granite4.0-h-32b": "Granite4.0-h-32B",
    "granite4.0-h-3b": "Granite4.0-h-3B"
}
sort_by = "model_size"
metric1 = "graph_compile_cached"
metric2 = "compiled_graph_sizes"
xlabel = ""
ylabel = "Loading Compiled Graphs Time (s)"
x2label = "Size (KB)"
y2label = "Loading Time (s)"
ylim_multiplier = 1.3
filename = script_dir / "figure6"
draw(model_names_map, iterations_path, sort_by, metric1, metric2, xlabel, ylabel, x2label, y2label, ylim_multiplier, filename)

# --- Figure 8 ---
model_names_map = {
    "llama2-13b-hf": "Llama2-13B",
    "falcon-7b": "Falcon-7B",
    "yi-6b": "Yi-6B",
    "qwen-0.5b": "Qwen-0.5B",
    "qwen-1.8b": "Qwen-1.8B",
    "qwen-7b": "Qwen-7B",
    "qwen-14b": "Qwen-14B",
    "qwen-14.3b": "Qwen-MoE-14.3B",
    "llama3-3b": "Llama3-3B",
    "deepseek-r1-distill-qwen-7b": "DeepSeek-R1-Distill-Qwen-7B",
}
sort_by = "model_size"
metric1 = "graph_capturing"
metric2 = "size"
xlabel = ""
ylabel = "Graph Capturing Time (s)"
x2label = "Model Size (B)"
y2label = "Capturing Time (s)"
ylim_multiplier = 1.5
filename = script_dir / "figure8"
draw(model_names_map, iterations_path, sort_by, metric1, metric2, xlabel, ylabel, x2label, y2label, ylim_multiplier, filename)

# --- Figure 17 ---
from predictor.run_predictor import predict

model_names_map = {
    "falcon-11b": "Falcon-11B",
    "mistral-7b": "Mistral-7B",
    "gemma-7b": "Gemma-7B",
    "llama3-3b": "Llama3-3B",
    "qwen-7b": "Qwen-7B",
}

results = predict(script_dir / "predictor_info/models", script_dir / "predictor_info/test_data.json")["validation"]

# Data
models = []
predicted = []
truth = []
diff = []
for x in results:
    label = x["label"]
    if label in model_names_map:
        models.append(model_names_map[label])
        predicted.append(x["predicted"])
        truth.append(x["truth"])
        diff.append(x["diff"])

actual = np.array(truth)
predictions = np.array(predicted)
mse = np.mean((actual - predictions) ** 2)
rmse = np.sqrt(mse)

x = np.arange(len(models))
width = 0.35  # bar width

fig, ax1 = plt.subplots(figsize=(8, 5))

# Bars: Predicted vs Truth
bars1 = ax1.bar(x - width/2, predicted, width, label="Predicted")
bars2 = ax1.bar(x + width/2, truth, width, label="Truth")

for bar in bars1 + bars2:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=11, color='white')

ax1.set_ylabel("vLLM's Startup Latency (s)", fontsize=15)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=15)
ax1.legend(fontsize=13)

plt.tight_layout()
plt.savefig(
    script_dir / "figure17.pdf",
    format="pdf",
    bbox_inches="tight",
    transparent=True
)