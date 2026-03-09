import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent

file = script_dir / "cpu_usage.json"
with open(file, "r") as f:
    usage_log = json.load(f)

samples = [x/10 for x in range(len(usage_log))]
core_data = list(zip(*[entry for entry in usage_log]))

core_data = np.array(core_data)  # shape: (cores, samples)

plt.figure(figsize=(10, 6))
im = plt.imshow(core_data, aspect="auto", cmap="viridis",
                extent=[samples[0], samples[-1], core_data.shape[0]-1, 0],
                vmin=0, vmax=100)

# Increase colorbar font sizes
cbar = plt.colorbar(im, label="CPU Usage (%)")
cbar.ax.set_ylabel("CPU Usage (%)", fontsize=15)
cbar.ax.tick_params(labelsize=12)

# Increase axis label and tick font sizes
plt.xlabel("Time (s)", fontsize=15)
plt.ylabel("Core ID", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig(
    script_dir / "figure12.pdf",
    format="pdf",
    bbox_inches="tight",
    transparent=True
)