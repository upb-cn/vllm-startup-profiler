import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)
from utils import *
from pathlib import Path

script_dir = Path(__file__).parent
iterations_path = script_dir / "iterations"

model_names_map = {
    "llama2-13b-hf": "Llama2-13B",
    "qwen-0.5b": "Qwen-0.5B",
    "qwen-1.8b": "Qwen-1.8B",
    "qwen-7b": "Qwen-7B",
    "qwen-4b": "Qwen-4B",
    "qwen-14b": "Qwen-14B",
    "yi-9b": "Yi-9B",
    "deepseek-r1-distill-llama-8b": "DeepSeek-R1-Distill-Llama-8B",
    "qwen-14.3b": "Qwen-MoE-14.3B",
    "deepseek-v2-lite-16b": "DeepSeek-V2-Lite-16B",
}
sort_by = "model_size"
metric1 = "kv_cache_profiling"
metric2 = "size"
xlabel = ""
ylabel = "KV Cache Profiling Time (s)"
x2label = "Model Size (B)"
y2label = "Profiling Time (s)"
ylim_multiplier = 1.7
excluded_models = ['Qwen-MoE-14.3B', 'DeepSeek-V2-Lite-16B']
filename = script_dir / "figure7"
draw(model_names_map, iterations_path, sort_by, metric1, metric2, xlabel, ylabel, x2label, y2label, ylim_multiplier, filename, excluded_labels=excluded_models)