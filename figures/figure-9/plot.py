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
}
sort_by = "model_size"
metric1 = "graph_capturing"
metric2 = "size"
xlabel = "Number of Batch Sizes"
ylabel = "Graph Capturing Time (s)"
x2label = "Number of Batch Sizes"
y2label = "Capturing Time (s)"
ylim_multiplier = 1.05
filename = script_dir / "figure9"
func = extract_batch_size

draw(model_names_map, iterations_path, sort_by, metric1, metric2, xlabel, ylabel, x2label, y2label, ylim_multiplier, filename, func)