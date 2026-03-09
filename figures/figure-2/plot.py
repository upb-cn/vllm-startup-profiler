import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path

script_dir = Path(__file__).parent

# --- Read results ---
file_path = script_dir / "avg_comparison_results.json"

with open(file_path, "r") as f:
    profiling_results = json.load(f)["data"]
    
    

# --- Step Information ---
steps = [
    ("Detect Platform (§2.1)", profiling_results["detect_platform"][0], "CPU"),
    ("LLM Imports (§2.1)",  profiling_results["llm_imports"][0], "CPU"),
    ("Get Model Info (§2.1)", profiling_results["get_model_info"][0], "CPU"),
    ("Worker Initialization (§2.1)", profiling_results["worker_init"][0], "CPU"),
    ("Tokenizer Initialization (§2.2)", profiling_results["tokenizer_init"][0], "CPU"),
    ("Model Initialization (§2.3)", profiling_results["model_init"][0], "CPU"),
    ("Loading Weights (§2.3)", profiling_results["load_weights"][0], "CPU"),
    ("Dynamo Transformation (§2.4)", profiling_results["dynamo_transform_time"][0], "CPU"),
    ("Load Compiled Graphs (§2.4)", profiling_results["graph_compile_cached"][0], "CPU"),
    ("KV Cache Profiling (§2.5)", profiling_results["kv_cache_profiling"][0], "GPU"),
    ("Graph Capturing (§2.6)", profiling_results["graph_capturing"][0], "GPU"),
]

# --- Assign Distinct Colors ---
step_colors = [
    "#4477AA", "#66CCEE", "#CCBB44", "#44AA99",
    "#117733", "#999933", "#CC6677", "#882255", 
    "#332288", "#AA4499", "#EE7733"
]

# --- Prepare Data ---
labels = [s[0] for s in steps]
times = [s[1] for s in steps]
types = [s[2] for s in steps]
total_time = sum(times)

# --- Compute cumulative bottoms for stacking ---
bottoms = np.cumsum([0] + times[:-1])

# --- Plot ---
fig, ax = plt.subplots(figsize=(1.5, 5.5))

# Draw stacked bar (with hatch for CPU)
bars = []
for i, (label, time, typ, bottom, color) in enumerate(zip(labels, times, types, bottoms, step_colors)):
    hatch = (
    '////' if typ == 'CPU' else
    ''
)
    b = ax.bar("Startup", time, bottom=bottom, color=color, edgecolor="black", hatch=hatch, linewidth=1.1)
    bars.append(b[0])
    if time >= 0.5:
        ax.text(0, bottom + time / 2, f"{time:.2f}s", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")

# --- Annotate Total ---
ax.text(0, total_time + 0.5, f"Total: {total_time:.2f}s", ha="center", va="bottom",
        fontsize=10, fontweight="bold")

# --- Styling ---
ax.set_ylim(0, total_time + 2)
ax.set_ylabel("Time (s)", fontsize=15)
ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax.tick_params(axis="y", labelsize=13)

# --- Legend for Steps ---
dep_patches = [
    Patch(facecolor='white', edgecolor='black', hatch='////', label='CPU-dominant'),
    Patch(facecolor='white', edgecolor='black', label='GPU-dominant')
]
dep_legend = ax.legend(
    dep_patches, [p.get_label() for p in dep_patches],
    loc='upper right',
    bbox_to_anchor=(2.5, 0.35),
    fontsize=11,
    title="Resource Dominance",
    frameon=True,
    borderaxespad=0.1,
    alignment="left"
)

# --- Legend for CPU/GPU dependency ---
ax.add_artist(dep_legend)
ax.legend(
    bars[::-1],
    labels[::-1],
    loc="upper right",
    bbox_to_anchor=(3.6, 1),
    fontsize=11,
    frameon=True,
    borderaxespad=0.5,
    alignment="left"
)

output_path = script_dir / "figure2.pdf"
plt.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        transparent=True
    )
plt.show()
