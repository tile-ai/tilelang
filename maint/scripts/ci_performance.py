import subprocess
import re
from tabulate import tabulate
import tilelang
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

tilelang.disable_cache()


def parse_output(output):
    data = {}
    for line in output.split('\n'):
        line = line.strip()
        m = re.match(r"\|\s*([^\|]+)\s*\|\s*([0-9\.]+)\s*\|", line)
        if m is not None:
            data[m.group(1)] = float(m.group(2))
    return data


output_v1 = subprocess.run(
    ['./tl/bin/python', '-c', 'import tilelang.tools.bench as b; b.bench_all()'],
    capture_output=True,
    text=True).stdout
output_v2 = subprocess.run(
    ['./tll/bin/python', '-c', 'import tilelang.tools.bench as b; b.bench_all()'],
    capture_output=True,
    text=True).stdout

data_v1 = parse_output(output_v1)
data_v2 = parse_output(output_v2)
table = []
for key in data_v1.keys():
    speedup = data_v1[key] / data_v2[key]
    table.append([key, data_v1[key], data_v2[key], speedup])
table.sort(key=lambda x: x[-1])

headers = ["File", "Original Latency", "Current Latency", "Speedup"]

with open("bench.md", "w") as f:
    f.write(
        tabulate(table, headers=headers, tablefmt="github", stralign="left", numalign="decimal"))
    f.write("\n")

df = pd.DataFrame(table, columns=headers)
df = df.sort_values("Speedup", ascending=False).reset_index(drop=True)
fig_width = max(0, len(df) * 0.35)
plt.figure(figsize=(fig_width, 8))
sns.set_theme(style="whitegrid", font_scale=0.9)
bar_colors = sns.color_palette("magma", len(df))
bars = plt.bar(range(len(df)), df["Speedup"], color=bar_colors, edgecolor="black")
top3_idx = df.nlargest(3, "Speedup").index
bot3_idx = df.nsmallest(3, "Speedup").index
label_idx = set(top3_idx.tolist() + bot3_idx.tolist())

for i, val in enumerate(df["Speedup"]):
    if i in label_idx:
        plt.text(
            i,
            val + 0.02,
            f"{val:.2f}x",
            ha="center",
            va="bottom",
            color="red",
            fontsize=8,
            fontweight="bold")

plt.xticks(range(len(df)), df["File"], rotation=70, ha='right', fontsize=12)
plt.ylabel("Current Speedup vs Original", fontsize=14)
plt.title("Current Speedup vs Original", fontsize=14, fontweight="bold")
plt.ylim(0, max(df["Speedup"]) * 1.2)
sns.despine()
plt.tight_layout()
plt.savefig("bench.png", dpi=300)
