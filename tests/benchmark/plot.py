import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

uplt.rc["font.name"] = "Fira Sans"

METHOD_ORDER = [
    "DiffDRR (float32)",
    "nanodrr (float32)",
    "nanodrr + compile (float32)",
    "nanodrr (bfloat16)",
    "nanodrr + compile (bfloat16)",
]

COLORS = {
    "light": {
        "DiffDRR (float32)": "#6C757D",
        "nanodrr (float32)": "#2E86AB",
        "nanodrr + compile (float32)": "#2E86AB",
        "nanodrr (bfloat16)": "#E63946",
        "nanodrr + compile (bfloat16)": "#E63946",
    },
    "dark": {
        "DiffDRR (float32)": "#ADB5BD",
        "nanodrr (float32)": "#5BB8D4",
        "nanodrr + compile (float32)": "#5BB8D4",
        "nanodrr (bfloat16)": "#FF6B74",
        "nanodrr + compile (bfloat16)": "#FF6B74",
    },
}
MARKERS = {
    "DiffDRR (float32)": "D",
    "nanodrr (float32)": "o",
    "nanodrr + compile (float32)": "s",
    "nanodrr (bfloat16)": "o",
    "nanodrr + compile (bfloat16)": "s",
}
LINESTYLES = {
    "DiffDRR (float32)": "-",
    "nanodrr (float32)": "-",
    "nanodrr + compile (float32)": (0, (1, 1)),
    "nanodrr (bfloat16)": "-",
    "nanodrr + compile (bfloat16)": (0, (1, 1)),
}


def format_label(method):
    base = method.replace(" + compile", "").replace(" (float32)", "").replace(" (bfloat16)", "")
    dtype = "fp32" if "float32" in method else "bf16" if "bfloat16" in method else ""
    compile_str = " + compile" if "+ compile" in method else ""
    return f"{base}\n({dtype}{compile_str})" if dtype else base


def get_values(df, method, pt_versions, col, err_col=None):
    vals, errs = [], []
    for ver in pt_versions:
        row = df[(df["pytorch_version"] == ver) & (df["name"] == method)]
        val = float(row[col].iloc[0]) if len(row) == 1 else np.nan
        err = float(row[err_col].iloc[0]) if (len(row) == 1 and err_col) else 0
        vals.append(val)
        errs.append(err)
    return vals, errs


def plot(df: pd.DataFrame, output: str) -> None:
    methods = [m for m in METHOD_ORDER if m in df["name"].unique()]
    pt_versions = sorted(df["pytorch_version"].unique(), key=lambda v: list(map(int, v.split("+")[0].split(".")[:2])))
    pt_labels = [v.split("+")[0].rsplit(".", 1)[0] for v in pt_versions]
    x = np.arange(len(pt_versions))

    for theme in ("light", "dark"):
        is_dark = theme == "dark"
        colors = COLORS[theme]

        fig, axs = uplt.subplots(ncols=2, sharex=True, sharey=False)

        for method in methods:
            kw = dict(color=colors[method], marker=MARKERS[method], linestyle=LINESTYLES[method], markersize=5)
            vals, errs = get_values(df, method, pt_versions, "fps_mean", "fps_std")
            axs[0].errorbar(x, vals, yerr=errs, label=format_label(method), **kw)
            vals, _ = get_values(df, method, pt_versions, "peak_reserved_mb")
            axs[1].errorbar(x, vals, **kw)

        axs[0].format(
            title="Rendering Speed (↑)",
            ylabel="Frames per Second [FPS]",
            xlabel="PyTorch Version",
            xticks=x,
            xticklabels=pt_labels,
        )
        axs[1].format(
            title="GPU Memory Usage (↓)",
            ylabel="Peak Memory Reserved [MB]",
            xlabel="PyTorch Version",
            xticks=x,
            xticklabels=pt_labels,
        )

        legend = fig.legend(loc="b", ncols=5, frameon=True)

        if is_dark:
            white = "white"
            fig.patch.set_facecolor("black")
            for ax in axs:
                ax.set_facecolor("black")
                ax.title.set_color(white)
                ax.xaxis.label.set_color(white)
                ax.yaxis.label.set_color(white)
                ax.tick_params(which="both", colors=white)
                for spine in ax.spines.values():
                    spine.set_edgecolor(white)
                ax.minorticks_on()
                ax.grid(which="major", color=white, linewidth=0.6, alpha=0.5)
                ax.grid(which="minor", visible=False)
            legend.get_frame().set_facecolor("black")
            legend.get_frame().set_edgecolor(white)
            for text in legend.get_texts():
                text.set_color(white)

        out = str(Path(output).with_stem(Path(output).stem + "_dark")) if is_dark else output
        fig.savefig(out, dpi=300, facecolor="black" if is_dark else None)
        print(f"{'Dark' if is_dark else 'Light'} figure saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument("--input", "-i", default=str(Path(__file__).parent / "benchmark.csv"))
    parser.add_argument("--output", "-o", default=str(Path(__file__).parent.parent.parent / "docs/assets/images/benchmark.png"))
    args = parser.parse_args()
    plot(pd.read_csv(args.input), args.output)


if __name__ == "__main__":
    main()
