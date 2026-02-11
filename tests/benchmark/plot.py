from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "Torch grid sample (float32)": "#4C72B0",
    "Torch grid sample + compile (float32)": "#4C72B0",
    "Torch grid sample (bfloat16)": "#DD8452",
    "Torch grid sample + compile (bfloat16)": "#DD8452",
    "DiffDRR (float32)": "#8C8C8C",
}

SHORT_LABELS = {
    "Torch grid sample (float32)": "fp32",
    "Torch grid sample + compile (float32)": "fp32\n+compile",
    "Torch grid sample (bfloat16)": "bf16",
    "Torch grid sample + compile (bfloat16)": "bf16\n+compile",
    "DiffDRR (float32)": "DiffDRR\n(fp32)",
}


def _parse_pytorch_major_minor(version_str: str) -> str:
    """'2.4.1+cu121' -> '2.4'"""
    base = version_str.split("+")[0]
    parts = base.split(".")
    return f"{parts[0]}.{parts[1]}"


def plot(df: pd.DataFrame, output: str) -> None:
    methods = list(SHORT_LABELS.keys())
    pt_versions = sorted(
        df["pytorch_version"].unique(),
        key=lambda v: list(map(int, v.split("+")[0].split(".")[:2])),
    )
    pt_labels = [_parse_pytorch_major_minor(v) for v in pt_versions]
    x = np.arange(len(pt_versions))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle(
        "nanodrr Benchmark — across PyTorch versions",
        fontsize=13,
        fontweight="bold",
    )

    panels = [
        {
            "ax": axes[0],
            "title": "Runtime",
            "col": "mean_us",
            "err_col": "std_us",
            "ylabel": "Time (μs)",
        },
        {
            "ax": axes[1],
            "title": "Throughput",
            "col": "fps_mean",
            "err_col": "fps_std",
            "ylabel": "FPS",
        },
        {
            "ax": axes[2],
            "title": "Peak Memory Reserved",
            "col": "peak_reserved_mb",
            "err_col": None,
            "ylabel": "Memory (MB)",
        },
    ]

    METHOD_MARKERS = {
        "Torch grid sample (float32)": "o",
        "Torch grid sample + compile (float32)": "s",
        "Torch grid sample (bfloat16)": "o",
        "Torch grid sample + compile (bfloat16)": "s",
        "DiffDRR (float32)": "D",
    }

    METHOD_LINESTYLES = {
        "Torch grid sample (float32)": "-",
        "Torch grid sample + compile (float32)": "--",
        "Torch grid sample (bfloat16)": "-",
        "Torch grid sample + compile (bfloat16)": "--",
        "DiffDRR (float32)": "-",
    }

    for panel in panels:
        ax = panel["ax"]
        for method in methods:
            vals, errs = [], []
            for ver in pt_versions:
                row = df[(df["pytorch_version"] == ver) & (df["name"] == method)]
                if len(row) == 1:
                    vals.append(float(row[panel["col"]].iloc[0]))
                    if panel["err_col"]:
                        errs.append(float(row[panel["err_col"]].iloc[0]))
                    else:
                        errs.append(0)
                else:
                    vals.append(np.nan)
                    errs.append(0)

            color = METHOD_COLORS.get(method, "#555555")
            marker = METHOD_MARKERS.get(method, "o")
            linestyle = METHOD_LINESTYLES.get(method, "-")
            label = SHORT_LABELS.get(method, method)

            ax.errorbar(
                x,
                vals,
                yerr=errs if panel["err_col"] else None,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=6,
                capsize=3,
                capthick=1,
                label=label,
                zorder=3,
            )

        ax.set_title(panel["title"])
        ax.set_ylabel(panel["ylabel"])
        ax.set_xticks(x)
        ax.set_xticklabels(pt_labels)
        ax.set_xlabel("PyTorch version")
        ax.grid(alpha=0.3, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

    # Single legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(methods),
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.savefig(output)
    print(f"Figure saved to {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument(
        "--input",
        "-i",
        default=str(Path(__file__).parent / "benchmark.csv"),
        help="Path to benchmark CSV (default: ./benchmark.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(Path(__file__).parent / "benchmark.png"),
        help="Output figure path (default: ./benchmark.png)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plot(df, args.output)


if __name__ == "__main__":
    main()
