import argparse
from pathlib import Path

import matplotlib.lines as mlines
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
    "nanodrr triton (float32)",
    "nanodrr triton + compile (float32)",
    "nanodrr triton (bfloat16)",
    "nanodrr triton + compile (bfloat16)",
]

# ColorBrewer Set2 (qualitative): nanodrr fp32/bf16, triton fp32/bf16
SET2 = {
    "nanodrr (float32)": "#66c2a5",
    "nanodrr + compile (float32)": "#66c2a5",
    "nanodrr (bfloat16)": "#8da0cb",
    "nanodrr + compile (bfloat16)": "#8da0cb",
    "nanodrr triton (float32)": "#fc8d62",
    "nanodrr triton + compile (float32)": "#fc8d62",
    "nanodrr triton (bfloat16)": "#e78ac3",
    "nanodrr triton + compile (bfloat16)": "#e78ac3",
}
COLORS = {
    "light": {"DiffDRR (float32)": "#6C757D", **SET2},
    "dark": {"DiffDRR (float32)": "#ADB5BD", **SET2},
}
MARKERS = {
    "DiffDRR (float32)": "D",
    "nanodrr (float32)": "o",
    "nanodrr + compile (float32)": "s",
    "nanodrr (bfloat16)": "o",
    "nanodrr + compile (bfloat16)": "s",
    "nanodrr triton (float32)": "^",
    "nanodrr triton + compile (float32)": "v",
    "nanodrr triton (bfloat16)": "^",
    "nanodrr triton + compile (bfloat16)": "v",
}
LINESTYLES = {
    "DiffDRR (float32)": "-",
    "nanodrr (float32)": "-",
    "nanodrr + compile (float32)": (0, (1, 1)),
    "nanodrr (bfloat16)": "-",
    "nanodrr + compile (bfloat16)": (0, (1, 1)),
    "nanodrr triton (float32)": (0, (3, 1)),
    "nanodrr triton + compile (float32)": (0, (5, 1)),
    "nanodrr triton (bfloat16)": (0, (3, 1)),
    "nanodrr triton + compile (bfloat16)": (0, (5, 1)),
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


def select_device(df: pd.DataFrame, device: str | None) -> pd.DataFrame:
    """Keep the rows of one device. CSVs written before the `device` column existed are CUDA-only."""
    if "device" not in df:
        return df
    devices = list(df["device"].dropna().unique())
    if device is None:
        device = "cuda" if "cuda" in devices else devices[0]
    if device not in devices:
        raise SystemExit(f"No rows for device {device!r} (have: {', '.join(map(str, devices))})")
    return df[df["device"] == device]


def plot(df: pd.DataFrame, output: str) -> None:
    methods = [m for m in METHOD_ORDER if m in df["name"].unique()]
    pt_versions = sorted(df["pytorch_version"].unique(), key=lambda v: list(map(int, v.split("+")[0].split(".")[:2])))
    pt_labels = [v.split("+")[0].rsplit(".", 1)[0] for v in pt_versions]
    x = np.arange(len(pt_versions))

    # CPU/MPS have no profiler pass (empty `gpu_us`), so fall back to wall-clock FPS
    has_gpu_time = df["fps_gpu"].notna().any()
    fps_col = "fps_gpu" if has_gpu_time else "fps"
    has_memory = df["peak_reserved_mb"].notna().any()

    for theme in ("light", "dark"):
        is_dark = theme == "dark"
        colors = COLORS[theme]

        fig, axs = uplt.subplots(ncols=2, sharex=True, sharey=False)

        handles = {}
        for method in methods:
            kw = {"color": colors[method], "marker": MARKERS[method], "linestyle": LINESTYLES[method], "markersize": 5}
            # GPU-time FPS is stable across runs; wall time is launch-bound
            vals, _ = get_values(df, method, pt_versions, fps_col)
            handles[method] = axs[0].errorbar(x, vals, label=format_label(method), **kw)
            vals, _ = get_values(df, method, pt_versions, "peak_reserved_mb")
            axs[1].errorbar(x, vals, **kw)

        speed_fmt = {"yscale": "log"}
        if has_gpu_time:  # fixed axis tuned for CUDA GPU-time FPS
            speed_fmt |= {
                "ylim": (100, 20000),
                "yticks": [100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                "yticklabels": ["100", "200", "500", "1,000", "2,000", "5,000", "10,000", "20,000"],
                "ytickminor": False,
            }
        axs[0].format(
            title="Rendering Speed (↑)",
            ylabel=f"Frames per Second [FPS, {'GPU' if has_gpu_time else 'wall'} time]",
            xlabel="PyTorch Version",
            xticks=x,
            xticklabels=pt_labels,
            **speed_fmt,
        )
        axs[1].format(
            title="GPU Memory Usage (↓)" if has_memory else "GPU Memory Usage (not recorded)",
            ylabel="Peak Memory Reserved [MB]",
            xlabel="PyTorch Version",
            xticks=x,
            xticklabels=pt_labels,
        )

        eager_row = [m for m in methods if "+ compile" not in m]
        compile_row = [c for m in eager_row if (c := m.replace(" (", " + compile (")) in handles]
        if len(eager_row) == len(compile_row) + 1:  # DiffDRR has no compiled counterpart
            blank = mlines.Line2D([], [], color="none")
            hs = [handles[m] for m in eager_row] + [blank] + [handles[m] for m in compile_row]
            ls = [format_label(m) for m in eager_row] + [" "] + [format_label(m) for m in compile_row]
            legend = fig.legend(hs, ls, loc="b", ncols=len(eager_row), order="C", frameon=True)
        else:
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
    parser.add_argument(
        "--output", "-o", default=str(Path(__file__).parent.parent.parent / "docs/assets/images/benchmark.png")
    )
    parser.add_argument("--device", default=None, help="Device rows to plot (default: cuda if present, else the first)")
    args = parser.parse_args()
    plot(select_device(pd.read_csv(args.input), args.device), args.output)


if __name__ == "__main__":
    main()
