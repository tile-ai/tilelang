"""Plot autoregressive model TPOT as grouped bars.

Reads ``bench_autoregressive.csv``, produces ``fig_autoregressive.(pdf|png)``.
"""

from __future__ import annotations
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

C_EAGER = "#BDBDBD"
C_IND = "#0072B2"
C_TL = "#D55E00"
BAR_KW = dict(edgecolor="black", linewidth=0.35)

MODEL_LABELS = {
    "llama2-7b": "LLaMA-2-7B",
    "qwen2.5-7b": "Qwen2.5-7B",
    "gemma-7b": "Gemma-7B",
}


def plot():
    rows = []
    with (RESULTS_DIR / "bench_autoregressive.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append(r)

    models = list(MODEL_LABELS.keys())
    data = {r["model"]: r for r in rows}

    eager = [float(data[m]["eager_ms"]) for m in models]
    ind = [float(data[m]["inductor_ms"]) for m in models]
    tl = [float(data[m]["tilelang_ms"]) for m in models]

    x = np.arange(len(models))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    ax.bar(x - width, eager, width, color=C_EAGER, label="Eager", **BAR_KW)
    ax.bar(x, ind, width, color=C_IND, label="Inductor", **BAR_KW)
    bars_tl = ax.bar(x + width, tl, width, color=C_TL, label="TileLang", **BAR_KW)

    # Speedup annotation on TileLang bars
    for i, b in enumerate(bars_tl):
        speedup = ind[i] / tl[i]
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("Decode TPOT on H100 (batch=1, prefill=128)", pad=8)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    top = max(eager) * 1.12
    ax.set_ylim(0, top)

    ax.legend(loc="upper right", ncol=3)

    fig.savefig(FIG_DIR / "fig_autoregressive.pdf")
    fig.savefig(FIG_DIR / "fig_autoregressive.png")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'fig_autoregressive.pdf'}")
    print(f"wrote {FIG_DIR / 'fig_autoregressive.png'}")


if __name__ == "__main__":
    plot()
