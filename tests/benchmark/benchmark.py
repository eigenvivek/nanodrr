import math
import platform
import statistics
import time
import warnings

import torch

from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.data import Subject
from nanodrr.drr import render

warnings.filterwarnings("ignore", message="dynamo_pgo force disabled")


def render_torch(*args):
    """Reference grid_sample backend (pinned so `auto` doesn't pick triton)."""
    return render(*args, backend="torch")


def render_triton(*args):
    """Fused Triton backend."""
    return render(*args, backend="triton")


def with_pose_grad(func):
    """Wrap a render function so one call is a forward pass plus a backward pass w.r.t. the pose."""

    def step(subject, k_inv, rt_inv, sdd, height, width):
        rt_inv = rt_inv.detach().clone().requires_grad_(True)
        out = func(subject, k_inv, rt_inv, sdd, height, width)
        out.sum().backward()
        return out

    return step


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def triton_available(device: str) -> bool:
    if device != "cuda":
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name()
    if device == "mps":
        return "Apple GPU (MPS)"
    return platform.processor() or platform.machine()


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def setup_data(image_path: str | None, label_path: str | None, device: str) -> tuple:
    """Load and prepare render inputs on `device`. `label_path` enables multi-class rendering."""
    if image_path is None:
        from nanodrr.data.demo import download_deepfluoro

        image_path, demo_label_path = download_deepfluoro()
        label_path = demo_label_path if label_path == "demo" else label_path
    elif label_path == "demo":
        raise SystemExit("--labels needs --label PATH when --image is given")
    subject = Subject.from_filepath(image_path, label_path)

    sdd = 1020.0
    delx = dely = 2.0
    x0 = y0 = 0.0
    height = width = 200

    k_inv = make_k_inv(sdd, delx, dely, x0, y0, height, width)
    rt_inv = make_rt_inv(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 850.0, 0.0]]),
        orientation="AP",
        isocenter=subject.isocenter,
    )
    sdd = torch.tensor([sdd])

    subject = subject.to(dtype=torch.float32, device=device)
    k_inv = k_inv.to(dtype=torch.float32, device=device)
    rt_inv = rt_inv.to(dtype=torch.float32, device=device)
    sdd = sdd.to(dtype=torch.float32, device=device)

    return subject, k_inv, rt_inv, sdd, height, width


def benchmark(
    func,
    *args,
    name: str = "Benchmark",
    device: str = "cuda",
    num_runs: int = 10,
    num_iterations: int = 100,
    warmup_iterations: int = 25,
    profile_iterations: int = 30,
) -> dict:
    """
    Benchmark a function on `device`.

    CUDA uses CUDA events for GPU timing, a profiler pass for pure GPU time and the
    caching-allocator memory stats. CPU and MPS use `time.perf_counter` with a
    synchronize around each run; `gpu_us` is NaN (no profiler pass) and memory is
    NaN on CPU. MPS reports the current (not peak) allocation.

    Args:
        func: Callable to benchmark
        *args: Arguments to pass to func
        name: Name of the benchmark (for printing)
        device: One of "cpu", "mps", "cuda"
        num_runs: Number of runs to average (default: 10)
        num_iterations: Number of iterations per run (default: 100)
        warmup_iterations: Number of warmup iterations (default: 25)
        profile_iterations: Iterations for the CUDA profiler pass measuring pure
            GPU time (default: 30)

    Returns:
        Dictionary with keys: wall_us, wall_std_us, gpu_us, fps, fps_std,
        fps_gpu, name, peak_allocated_mb, peak_reserved_mb, delta_allocated_mb
    """
    nan = float("nan")
    is_cuda = device == "cuda"

    # Warmup
    for _ in range(warmup_iterations):
        func(*args)
    synchronize(device)

    # Record memory baseline after warmup (captures compile overhead separately)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
    elif device == "mps":
        mem_before = torch.mps.current_allocated_memory()

    times = []
    for _ in range(num_runs):
        synchronize(device)
        if is_cuda:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
        else:
            start = time.perf_counter()
        for _ in range(num_iterations):
            func(*args)
        if is_cuda:
            t1.record()
            torch.cuda.synchronize()
            # elapsed_time() returns milliseconds; convert to microseconds
            times.append(t0.elapsed_time(t1) / num_iterations * 1000)
        else:
            synchronize(device)
            times.append((time.perf_counter() - start) / num_iterations * 1e6)

    # Collect memory stats
    if is_cuda:
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        delta_allocated = torch.cuda.memory_allocated() - mem_before
    elif device == "mps":
        peak_allocated = torch.mps.current_allocated_memory()
        peak_reserved = torch.mps.driver_allocated_memory()
        delta_allocated = peak_allocated - mem_before
    else:
        peak_allocated = peak_reserved = delta_allocated = nan

    # Median wall time: contention only slows runs down, so the median tracks
    # the uncontended machine better than the mean
    wall_us = statistics.median(times)
    wall_std = statistics.pstdev(times)
    fps = 1e6 / wall_us
    fps_values = [1e6 / t for t in times]
    fps_std = statistics.pstdev(fps_values)

    # GPU time from a profiler pass; the gap between wall_us and gpu_us is
    # CPU launch overhead
    gpu_us = nan
    if is_cuda:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(profile_iterations):
                func(*args)
            torch.cuda.synchronize()
        gpu_us = (
            sum(getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) for e in prof.key_averages())
            / profile_iterations
        )
    fps_gpu = 1e6 / gpu_us if gpu_us > 0 else nan

    print(f"\n{name} [{device}]:")
    print(
        f"  wall: {wall_us:,.0f} μs (median of {num_runs} runs ± {wall_std:,.0f} μs, "
        f"{num_iterations:,} loops each) → {fps:,.0f} FPS"
    )
    if is_cuda:
        print(f"  gpu:  {gpu_us:,.0f} μs → {fps_gpu:,.0f} FPS (launch overhead: {wall_us - gpu_us:+,.0f} μs)")
    print(
        f"  Peak memory allocated: {peak_allocated / 1024**2:,.1f} MB | "
        f"Peak memory reserved: {peak_reserved / 1024**2:,.1f} MB | "
        f"Delta allocated: {delta_allocated / 1024**2:+,.1f} MB"
    )

    return {
        "wall_us": wall_us,
        "wall_std_us": wall_std,
        "gpu_us": gpu_us,
        "fps": fps,
        "fps_std": fps_std,
        "fps_gpu": fps_gpu,
        "name": name,
        "peak_allocated_mb": peak_allocated / 1024**2,
        "peak_reserved_mb": peak_reserved / 1024**2,
        "delta_allocated_mb": delta_allocated / 1024**2,
    }


def run_device(device: str, args) -> list[dict]:
    """Benchmark every configuration supported on `device`; a failing one is skipped, not fatal."""
    # CPU renders are orders of magnitude slower, so use far fewer iterations
    cpu = device == "cpu"
    bench_kw = {
        "device": device,
        "num_runs": args.runs or (5 if cpu else 10),
        "num_iterations": args.iterations or (3 if cpu else 100),
        "warmup_iterations": 2 if cpu else 25,
    }
    results = []

    task = "forward+backward" if args.grad else "forward"

    def attempt(name, func, *inputs):
        try:
            if args.grad:
                func = with_pose_grad(func)
            results.append({**benchmark(func, *inputs, name=name, **bench_kw), "n_classes": n_classes, "task": task})
        except Exception as e:  # noqa: BLE001  # unsupported dtype/op on this device, compile failure, ...
            print(f"\n{name} [{device}]: skipped ({type(e).__name__}: {str(e).splitlines()[0][:120]})")

    def compiled(func):
        torch._dynamo.reset()
        # CUDA graphs ("reduce-overhead") only exist on CUDA
        mode = "reduce-overhead" if device == "cuda" else None
        return torch.compile(func, mode=mode, fullgraph=True)

    label_path = args.label or ("demo" if args.labels else None)
    inputs = setup_data(args.image, label_path, device)
    n_classes = inputs[0].n_classes
    print(f"Task: {task}, {n_classes} class{'es' if n_classes != 1 else ''}")
    use_triton = triton_available(device) and not args.no_triton

    # Compile configuration
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._inductor.config.force_disable_caches = True

    attempt("nanodrr (float32)", render_torch, *inputs)
    if not args.no_compile:
        attempt("nanodrr + compile (float32)", compiled(render_torch), *inputs)
    if use_triton:
        # Triton float32 runs before the in-place bfloat16 cast below
        attempt("nanodrr triton (float32)", render_triton, *inputs)
        if not args.no_compile:
            attempt("nanodrr triton + compile (float32)", compiled(render_triton), *inputs)

    # bfloat16
    subject, k_inv, rt_inv, sdd, height, width = inputs
    inputs_bf16 = (subject.bfloat16(), k_inv.bfloat16(), rt_inv.bfloat16(), sdd.bfloat16(), height, width)
    attempt("nanodrr (bfloat16)", render_torch, *inputs_bf16)
    if not args.no_compile:
        attempt("nanodrr + compile (bfloat16)", compiled(render_torch), *inputs_bf16)
    if use_triton:
        attempt("nanodrr triton (bfloat16)", render_triton, *inputs_bf16)
        if not args.no_compile:
            attempt("nanodrr triton + compile (bfloat16)", compiled(render_triton), *inputs_bf16)

    # DiffDRR baseline (float32), optional: it is not a nanodrr dependency and may not support every device
    if (label_path is not None or args.grad) and not args.no_diffdrr:
        print("\nDiffDRR: skipped (the baseline is single-class and forward-only here)")
    elif not args.no_diffdrr:
        try:
            from diffdrr.data import read
            from diffdrr.drr import DRR
            from diffdrr.pose import convert
        except ImportError:
            print("\nDiffDRR: skipped (not installed; run with `--with diffdrr`)")
        else:
            try:
                image_path = args.image or setup_image_path()
                drr = DRR(read(image_path), sdd=1020.0, height=200, delx=2.0, renderer="trilinear").to(device)
                pose = convert(
                    torch.tensor([[0.0, 0.0, 0.0]]),
                    torch.tensor([[0.0, 850.0, 0.0]]),
                    parameterization="euler_angles",
                    convention="ZXY",
                ).to(device)

                attempt("DiffDRR (float32)", drr, pose)
            except Exception as e:  # noqa: BLE001
                print(f"\nDiffDRR [{device}]: skipped ({type(e).__name__}: {str(e).splitlines()[0][:120]})")

    return results


def setup_image_path() -> str:
    from nanodrr.data.demo import download_deepfluoro

    return download_deepfluoro()[0]


def main():
    import argparse
    import csv
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="benchmark_results.csv", help="CSV output path")
    parser.add_argument(
        "--device",
        "-d",
        nargs="+",
        default=None,
        choices=["cpu", "mps", "cuda", "all"],
        help="Device(s) to benchmark (default: the best available; 'all' = every available device)",
    )
    parser.add_argument("--image", default=None, help="CT volume (default: the DeepFluoro demo subject)")
    parser.add_argument("--label", default=None, help="Labelmap for multi-class rendering")
    parser.add_argument("--labels", action="store_true", help="Multi-class rendering with the demo subject's labelmap")
    parser.add_argument("--grad", action="store_true", help="Time forward + backward w.r.t. the pose, not forward only")
    parser.add_argument("--runs", type=int, default=None, help="Timed runs per configuration")
    parser.add_argument("--iterations", type=int, default=None, help="Renders per timed run")
    parser.add_argument("--no-compile", action="store_true", help="Skip the torch.compile configurations")
    parser.add_argument("--no-triton", action="store_true", help="Skip the Triton configurations")
    parser.add_argument("--no-diffdrr", action="store_true", help="Skip the DiffDRR baseline")
    args = parser.parse_args()

    available = available_devices()
    requested = args.device or [available[-1]]
    devices = available if "all" in requested else list(dict.fromkeys(requested))
    for d in devices:
        if d not in available:
            parser.error(f"device {d!r} is not available (have: {', '.join(available)})")

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    try:
        import triton

        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Triton: not installed")

    all_results = []
    for device in devices:
        print(f"\n=== {device} ({device_name(device)}) ===")
        all_results += [{**r, "device": device, "device_name": device_name(device)} for r in run_device(device, args)]

    # Save results to CSV
    try:
        import triton as _triton

        triton_version = _triton.__version__
    except ImportError:
        triton_version = ""

    meta = {
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "triton_version": triton_version,
    }

    fieldnames = [
        "pytorch_version",
        "cuda_version",
        "triton_version",
        "device",
        "device_name",
        "task",
        "n_classes",
        "name",
        "wall_us",
        "wall_std_us",
        "gpu_us",
        "fps",
        "fps_std",
        "fps_gpu",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "delta_allocated_mb",
    ]

    def fmt(x):
        return "" if math.isnan(x) else f"{x:.1f}"

    write_header = not os.path.exists(args.output)
    if not write_header:
        with open(args.output, newline="") as f:
            if next(csv.reader(f), None) != fieldnames:
                parser.error(f"{args.output} has a different column layout; choose another --output")
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in all_results:
            writer.writerow(
                {
                    **meta,
                    "device": r["device"],
                    "device_name": r["device_name"],
                    "task": r["task"],
                    "n_classes": r["n_classes"],
                    "name": r["name"],
                    **{
                        k: fmt(r[k])
                        for k in (
                            "wall_us",
                            "wall_std_us",
                            "gpu_us",
                            "fps",
                            "fps_std",
                            "fps_gpu",
                            "peak_allocated_mb",
                            "peak_reserved_mb",
                            "delta_allocated_mb",
                        )
                    },
                }
            )
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
