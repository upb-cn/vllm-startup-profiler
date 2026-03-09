import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)
from utils import *
from pathlib import Path

script_dir = Path(__file__).parent
iterations_path = script_dir / "iterations"

model_names_map = {
    "llama2-7b-hf": "Llama2-7B",
    "llama2-13b-hf": "Llama2-13B",
    "yi-6b": "Yi-6B",
    "yi-9b": "Yi-9B",
    "qwen-4b": "Qwen-4B",
    "qwen-14b": "Qwen-14B",
    "qwen-14.3b": "Qwen-MoE-14.3B",
    "llama3-3b": "Llama3-3B",
    "deepseek-v2-lite-16b": "DeepSeek-V2-Lite-16B",
    "gpt-oss-20b": "GPT-OSS-20B"
}

sort_by = "model_size"
metric1 = "graph_compile_general_shape"
metric2 = "compiled_graph_sizes"
xlabel = ""
ylabel = "Storing Compiled Graphs Time (s)"
x2label = "Size (KB)"
y2label = "Storing Time (s)"
ylim_multiplier = 1.3
filename = script_dir / "figure15"
draw(model_names_map, iterations_path, sort_by, metric1, metric2, xlabel, ylabel, x2label, y2label, ylim_multiplier, filename)