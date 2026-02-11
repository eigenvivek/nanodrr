import warnings

import torch

from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.data import Subject
from nanodrr.drr import render

warnings.filterwarnings("ignore", message="dynamo_pgo force disabled")


def setup_data(image_path: str, label_path: str | None = None) -> tuple:
    """Load and prepare render inputs."""
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

    # Move to cuda
    subject = subject.to(dtype=torch.float32, device="cuda")
    k_inv = k_inv.to(dtype=torch.float32, device="cuda")
    rt_inv = rt_inv.to(dtype=torch.float32, device="cuda")
    sdd = sdd.to(dtype=torch.float32, device="cuda")

    return subject, k_inv, rt_inv, sdd, height, width


def benchmark(
    func,
    *args,
    name: str = "Benchmark",
    num_runs: int = 10,
    num_iterations: int = 100,
    warmup_iterations: int = 25,
) -> dict:
    """
    Benchmark a function using CUDA events for accurate GPU timing.

    Args:
        func: Callable to benchmark
        *args: Arguments to pass to func
        name: Name of the benchmark (for printing)
        num_runs: Number of runs to average (default: 10)
        num_iterations: Number of iterations per run (default: 100)
        warmup_iterations: Number of warmup iterations (default: 25)

    Returns:
        Dictionary with keys: mean, std, name
    """
    # Warmup
    for _ in range(warmup_iterations):
        func(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(num_iterations):
            func(*args)
        t1.record()
        torch.cuda.synchronize()
        # elapsed_time() returns milliseconds; convert to microseconds
        times.append(t0.elapsed_time(t1) / num_iterations * 1000)

    mean = sum(times) / len(times)
    std = (sum((x - mean) ** 2 for x in times) / len(times)) ** 0.5

    print(f"\n{name}:")
    print(
        f"  {mean:,.0f} μs ± {std:,.0f} μs per loop "
        f"(mean ± std. dev. of {num_runs} runs, {num_iterations:,} loops each)"
    )

    return {"mean": mean, "std": std, "name": name}


def main():
    import sys

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    try:
        import triton

        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Triton: not installed")

    # Setup
    subject, k_inv, rt_inv, sdd, height, width = setup_data("data/image.nii.gz", None)

    # Compile configuration
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._inductor.config.force_disable_caches = True

    # Benchmark float32
    benchmark(
        render,
        subject,
        k_inv,
        rt_inv,
        sdd,
        height,
        width,
        name="Torch grid sample (float32)",
    )

    # Benchmark float32 + compile
    torch._dynamo.reset()
    render_compiled = torch.compile(render, mode="reduce-overhead", fullgraph=True)
    benchmark(
        render_compiled,
        subject,
        k_inv,
        rt_inv,
        sdd,
        height,
        width,
        name="Torch grid sample + compile (float32)",
    )

    # Benchmark bfloat16
    torch._dynamo.reset()
    subject_bf16 = subject.bfloat16()
    k_inv_bf16 = k_inv.bfloat16()
    rt_inv_bf16 = rt_inv.bfloat16()
    sdd_bf16 = sdd.bfloat16()

    benchmark(
        render,
        subject_bf16,
        k_inv_bf16,
        rt_inv_bf16,
        sdd_bf16,
        height,
        width,
        name="Torch grid sample (bfloat16)",
    )

    # Benchmark bfloat16 + compile
    torch._dynamo.reset()
    render_bf16_compiled = torch.compile(render, mode="reduce-overhead", fullgraph=True)
    benchmark(
        render_bf16_compiled,
        subject_bf16,
        k_inv_bf16,
        rt_inv_bf16,
        sdd_bf16,
        height,
        width,
        name="Torch grid sample + compile (bfloat16)",
    )

    # DiffDRR baseline (float32)
    from diffdrr.data import read
    from diffdrr.drr import DRR
    from diffdrr.pose import convert

    diffdrr_subject = read("data/image.nii.gz")
    drr = DRR(diffdrr_subject, sdd=1020.0, height=200, delx=2.0, renderer="trilinear").cuda()
    pose = convert(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 850.0, 0.0]]),
        parameterization="euler_angles",
        convention="ZXY",
    ).cuda()

    benchmark(
        drr,
        pose,
        name="DiffDRR (float32)",
    )


if __name__ == "__main__":
    main()
