import contextlib
import subprocess
import re
import os
import json
from tabulate import tabulate
import pandas as pd
import numpy as np
import textwrap

try:
    import tilelang

    tilelang.disable_cache()
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
except Exception:
    tilelang = None

OLD_PYTHON = os.environ.get("OLD_PYTHON", "./old/bin/python")
NEW_PYTHON = os.environ.get("NEW_PYTHON", "./new/bin/python")
OUT_MD = os.environ.get("PERF_REGRESSION_MD", "regression_result.md")
OUT_PNG = os.environ.get("PERF_REGRESSION_PNG", "regression_result.png")
_RESULTS_JSON_PREFIX = "__TILELANG_PERF_RESULTS_JSON__="


def parse_output(output):
    for line in output.splitlines():
        if line.startswith(_RESULTS_JSON_PREFIX):
            return json.loads(line[len(_RESULTS_JSON_PREFIX) :])

    # Fallback to regex parsing
    data = {}
    for line in output.split("\n"):
        line = line.strip()
        m = re.match(r"\|\s*([^\|]+)\s*\|\s*([0-9\.]+)\s*\|", line)
        if m is not None:
            with contextlib.suppress(ValueError):
                data[m.group(1)] = float(m.group(2))
    return data


def run_cmd(cmd, env=None):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    # Don't capture stderr so that tqdm progress bar is visible
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True, env=full_env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}")
    return p.stdout


def draw(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if df is None or len(df) == 0:
        return

    # ---- copy + sanitize ----
    df = df.copy()
    df["Speedup"] = pd.to_numeric(df["Speedup"], errors="coerce")
    df = df.dropna(subset=["Speedup"])

    # categorize
    df["Performance"] = np.where(df["Speedup"] >= 1.0, "Improved", "Regressed")
    df["DeltaPct"] = (df["Speedup"] - 1.0) * 100.0

    # sort: worst regressions at top? (common for dashboards)
    # If you prefer best-to-worst, change ascending=False
    df = df.sort_values("Speedup", ascending=True).reset_index(drop=True)

    # ---- style ----
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    n = len(df)
    # height: ~0.35 inch per row + margins, with a sensible cap/floor
    fig_h = min(max(6.0, 0.35 * n + 2.2), 22.0)
    fig_w = 14.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # palette
    colors = {"Improved": "#2ecc71", "Regressed": "#e74c3c"}
    bar_colors = df["Performance"].map(colors).tolist()

    # wrap long labels (optional)
    def wrap_label(s: str, width: int = 42) -> str:
        return "\n".join(textwrap.wrap(str(s), width=width)) if len(str(s)) > width else str(s)

    ylabels = [wrap_label(x) for x in df["File"].tolist()]
    y = np.arange(n)

    # bars
    ax.barh(y, df["Speedup"].values, color=bar_colors, edgecolor="black", linewidth=0.4, height=0.72)

    # baseline at 1.0x
    ax.axvline(1.0, linestyle="--", linewidth=1.4, alpha=0.85)

    # grid
    ax.xaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

    # y ticks
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)

    # x limits with padding (ensure 1.0 included)
    x_min = float(df["Speedup"].min())
    x_max = float(df["Speedup"].max())
    pad = max(0.02, (x_max - x_min) * 0.12)
    left = min(1.0, x_min) - pad
    right = max(1.0, x_max) + pad
    ax.set_xlim(left, right)

    # annotate each bar
    for i, (sx, dp) in enumerate(zip(df["Speedup"].values, df["DeltaPct"].values)):
        label = f"{sx:.3f}x ({dp:+.2f}%)"
        # place to right for improved, left for regressed (near bar end)
        if sx >= 1.0:
            ax.text(sx + 0.003, i, label, va="center", ha="left", fontsize=9)
        else:
            ax.text(sx - 0.003, i, label, va="center", ha="right", fontsize=9)

    # labels & title
    ax.set_xlabel("Speedup Ratio (New / Old)")
    ax.set_ylabel("Benchmark File")
    ax.set_title("Performance Regression Analysis")

    # legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=colors["Improved"], edgecolor="black", label="Improved (>= 1.0x)"),
        Patch(facecolor=colors["Regressed"], edgecolor="black", label="Regressed (< 1.0x)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True)

    # summary box
    num_improved = int((df["Performance"] == "Improved").sum())
    num_regressed = int((df["Performance"] == "Regressed").sum())
    best = df.iloc[df["Speedup"].idxmax()]
    worst = df.iloc[df["Speedup"].idxmin()]
    summary = (
        f"Items: {n}\n"
        f"Improved: {num_improved}\n"
        f"Regressed: {num_regressed}\n"
        f"Best:  {best['File']}  {best['Speedup']:.3f}x\n"
        f"Worst: {worst['File']}  {worst['Speedup']:.3f}x"
    )
    ax.text(
        0.99,
        0.01,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.3", alpha=0.9),
    )

    # clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    print(f"Saving plot to {OUT_PNG} ({fig_w:.1f}x{fig_h:.1f} inches)")
    fig.savefig(OUT_PNG, bbox_inches="tight")
    # Optional: also save SVG
    # fig.savefig(OUT_PNG.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)


env = {"TL_PERF_REGRESSION_FORMAT": "json"}
print("Running regression on OLD version...")
output_v1 = run_cmd([OLD_PYTHON, "-c", "import tilelang.testing.perf_regression as pr; pr.regression_all()"], env=env)
print("Running regression on NEW version...")
output_v2 = run_cmd([NEW_PYTHON, "-c", "import tilelang.testing.perf_regression as pr; pr.regression_all()"], env=env)

data_v1 = parse_output(output_v1)
data_v2 = parse_output(output_v2)

common_keys = sorted(set(data_v1) & set(data_v2))
if not common_keys:
    print("No common entries between old and new versions")
    # Write empty file or message
    with open(OUT_MD, "w") as f:
        f.write("No common benchmarks found between old and new versions.\n")
    exit(0)

table = []
for key in common_keys:
    if data_v2[key] == 0:
        speedup = 0.0
    else:
        speedup = data_v1[key] / data_v2[key]
    table.append([key, data_v1[key], data_v2[key], speedup])

if not table:
    raise RuntimeError("All results are invalid")

table.sort(key=lambda x: x[-1])

headers = ["File", "Original Latency", "Current Latency", "Speedup"]

with open(OUT_MD, "w") as f:
    f.write(tabulate(table, headers=headers, tablefmt="github", stralign="left", numalign="decimal"))
    f.write("\n")

df = pd.DataFrame(table, columns=headers)
df = df.sort_values("Speedup", ascending=False).reset_index(drop=True)
draw(df)
