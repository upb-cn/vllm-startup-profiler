import matplotlib.pyplot as plt
import os
import re
from pathlib import Path

script_dir = Path(__file__).parent

def version_key(item):
    version_str = item['label'].split('_')[1].replace('.txt', '')
    return tuple(map(int, version_str.split('.')))


data = []
for file in os.listdir(script_dir / "iterations"):
    with open(script_dir / "iterations" / file, "r") as f:
        for line in f.readlines():
            if "initialize_engine" in line:
                pattern = r"initialize_engine took ([\d.]+) seconds"
                match = re.search(pattern, line)
                s = float(match.group(1))
                data.append({
                    "label": file,
                    "time": s
                })
           
data = sorted(data, key=version_key)

labels = [item["label"].split("_")[-1].split(".txt")[0] for item in data]
values = [round(item["time"], 2) for item in data]
release_dates = ["Feb 24", "Apr 24", "Jun 24", "Sep 24", "Jan 25", "Mar 25", "May 25", "Jul 25", "Oct 25"]

plt.figure(figsize=(8.5, 5))
bars = plt.bar([f"{l}\n{release_dates[i]}\n" for i,l in enumerate(labels)], values)
plt.ylabel("Start Up Time (s)", fontsize=15)
plt.xlabel("vLLM Version", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# Add value labels on top of each bar
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{value}', ha='center', va='bottom', fontsize=10, color="white")

plt.savefig(
    script_dir / "figure1.pdf",
    format="pdf",
    bbox_inches="tight",
    transparent=True
)
plt.show()