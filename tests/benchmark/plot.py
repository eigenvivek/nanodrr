import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

uplt.rc["font.name"] = "Fira Sans"

METHOD_COLORS = {
    "nanodrr (float32)": "#2E86AB",
    "nanodrr + compile (float32)": "#2E86AB",
    "nanodrr (bfloat16)": "#E63946",
    "nanodrr + compile (bfloat16)": "#E63946",
    "DiffDRR (float32)": "#6C757D",
}

METHOD_MARKERS = {
    "DiffDRR (float32)": "D",
    "nanodrr (float32)": "o",
    "nanodrr + compile (float32)": "s",
    "nanodrr (bfloat16)": "o",
    "nanodrr + compile (bfloat16)": "s",
}

METHOD_LINESTYLES = {
    "nanodrr (float32)": "-",
    "nanodrr + compile (float32)": (0, (1, 1)),
    "nanodrr (bfloat16)": "-",
    "nanodrr + compile (bfloat16)": (0, (1, 1)),
    "DiffDRR (float32)": "-",
}

METHOD_ORDER = [
    "DiffDRR (float32)",
    "nanodrr (float32)",
    "nanodrr + compile (float32)",
    "nanodrr (bfloat16)",
    "nanodrr + compile (bfloat16)",
]


def format_label(method):
    """Convert 'nanodrr + compile (float32)' -> 'nanodrr\n(fp32 + compile)'"""
    base = method.replace(" + compile", "").replace(" (float32)", "").replace(" (bfloat16)", "")
    dtype = "fp32" if "float32" in method else "bf16" if "bfloat16" in method else ""
    compile_str = " + compile" if "+ compile" in method else ""
    return f"{base}\n({dtype}{compile_str})" if dtype else base


def get_values(df, method, pt_versions, col, err_col=None):
    """Extract values and errors for a method across versions"""
    vals, errs = [], []
    for ver in pt_versions:
        row = df[(df["pytorch_version"] == ver) & (df["name"] == method)]
        if len(row) == 1:
            vals.append(float(row[col].iloc[0]))
            errs.append(float(row[err_col].iloc[0]) if err_col else 0)
        else:
            vals.append(np.nan)
            errs.append(0)
    return vals, errs


def plot(df: pd.DataFrame, output: str) -> None:
    methods = [m for m in METHOD_ORDER if m in df["name"].unique()]
    pt_versions = sorted(df["pytorch_version"].unique(), key=lambda v: list(map(int, v.split("+")[0].split(".")[:2])))
    pt_labels = [v.split("+")[0].rsplit(".", 1)[0] for v in pt_versions]
    x = np.arange(len(pt_versions))

    fig, axs = uplt.subplots(ncols=2, sharex=True, sharey=False)

    # Left: FPS with error bars
    for method in methods:
        vals, errs = get_values(df, method, pt_versions, "fps_mean", "fps_std")
        axs[0].errorbar(
            x,
            vals,
            yerr=errs,
            label=format_label(method),
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            markersize=5,
        )

    axs[0].format(
        title="Rendering Speed (↑)",
        ylabel="Frames per Second [FPS]",
        xlabel="PyTorch Version",
        xticks=x,
        xticklabels=pt_labels,
    )

    # Right: Memory (no labels or error bars)
    for method in methods:
        vals, _ = get_values(df, method, pt_versions, "peak_reserved_mb")
        axs[1].errorbar(
            x,
            vals,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            markersize=5,
        )

    axs[1].format(
        title="GPU Memory Usage (↓)",
        ylabel="Peak Memory Reserved [MB]",
        xlabel="PyTorch Version",
        xticks=x,
        xticklabels=pt_labels,
    )

    fig.legend(loc="b", ncols=5, frameon=True)
    fig.savefig(output, dpi=300)
    print(f"Figure saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument("--input", "-i", default=str(Path(__file__).parent / "benchmark.csv"))
    parser.add_argument("--output", "-o", default=str(Path(__file__).parent / "benchmark.png"))
    args = parser.parse_args()

    plot(pd.read_csv(args.input), args.output)


if __name__ == "__main__":
    main()
