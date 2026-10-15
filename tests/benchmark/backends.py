"""Compare rendering backends on every (device, backend) pair this machine supports.

Pairs: torch on cpu/mps/cuda, plus the fused kernels (metal on MPS, triton on CUDA).
Reports accuracy against an all-float32 torch render on the same device, speed, and working-set memory
(peak extra device memory while one render runs, on top of the resident volume).

Usage: python tests/benchmark/backends.py [--grad {pose,volume}] [--labels | --soft-labels] [--size 200 400] [--samples 500] [--batch 1]
"""

import argparse
import copy
import functools
import statistics
import time

import torch
from benchmark import PeakMemory, available_devices, device_name, setup_data, synchronize

from nanodrr.camera import make_k_inv
from nanodrr.drr import render
from nanodrr.drr._fused import DEVICE_BACKENDS, available


def available_backends(device: str) -> list[str]:
    """Backends runnable on `device`: torch everywhere, plus the fused kernel for that GPU."""
    backends = ["torch"]
    fused = DEVICE_BACKENDS.get(device)
    if fused and available(fused):
        backends.append(fused)
    return backends


def pose_grad(args, backend, soft_labels=False):
    """Gradient of the summed render w.r.t. the pose (one forward + backward pass)."""
    subject, k_inv, rt_inv, *rest = args
    rt_inv = rt_inv.detach().clone().requires_grad_(True)
    render(subject, k_inv, rt_inv, *rest, backend=backend, soft_labels=soft_labels).sum().backward()
    return rt_inv.grad[:, :3]  # the homogeneous row's gradient is backend-specific


def volume_grad(args, backend, soft_labels=False):
    """Gradient of the summed render w.r.t. the volume (the subject must have `convert_to_mu` off)."""
    subject = args[0]
    subject._image_hu.grad = None
    render(*args, backend=backend, soft_labels=soft_labels).sum().backward()
    return subject._image_hu.grad


GRADS = {"pose": pose_grad, "volume": volume_grad}


def working_set_mb(func, device: str) -> float:
    """Peak extra device memory (MB) allocated by one call of `func`."""
    synchronize(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        func()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / 1024**2
    with PeakMemory(device) as tracker:
        func()
    synchronize(device)
    return tracker.peak / 1024**2


def time_us(func, device: str, runs: int, iters: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        func()
    times = []
    for _ in range(runs):
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            func()
        synchronize(device)
        times.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(times), statistics.pstdev(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None)
    p.add_argument("--labels", action="store_true", help="Multi-class rendering with the demo labelmap")
    p.add_argument(
        "--soft-labels", action="store_true", help="Trilinear (partial-volume) label assignment; implies --labels"
    )
    p.add_argument("--size", type=int, nargs="+", default=[100, 200, 400], help="Square detector sizes")
    p.add_argument("--samples", type=int, default=500)
    p.add_argument(
        "--grad",
        nargs="?",
        const="pose",
        choices=["pose", "volume"],
        help="Time forward + backward w.r.t. the pose (default) or the volume",
    )
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32", help="Volume dtype")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", nargs="+", default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--cpu-iterations", type=int, default=2, help="CPU is far slower, so loop less")
    a = p.parse_args()

    for device in a.device or available_devices():
        subject, _, rt_inv, sdd, _, _ = setup_data(a.image, "demo" if a.labels or a.soft_labels else None, device)
        dt = getattr(torch, a.dtype)
        # Errors are measured against an all-float32 torch render, so they include
        # each backend's low-precision error; `--dtype` casts the volume, labels and geometry
        ref_subject = copy.deepcopy(subject)
        if a.dtype != "float32":
            subject, rt_inv_lp, sdd_lp = subject.to(dt), rt_inv.to(dt), sdd.to(dt)
        else:
            rt_inv_lp, sdd_lp = rt_inv, sdd
        if a.grad == "volume":  # differentiate the raw volume, not the HU -> mu conversion
            for sub in (subject, ref_subject):
                sub.convert_to_mu = False
                sub._image_hu = sub._image_hu.detach().requires_grad_(True)
        rt_inv, sdd = rt_inv.expand(a.batch, -1, -1), sdd.expand(a.batch)
        rt_inv_lp, sdd_lp = rt_inv_lp.expand(a.batch, -1, -1), sdd_lp.expand(a.batch)
        print(
            f"\n### {device} ({device_name(device)}): volume {tuple(subject.image.shape[2:])}, "
            f"{subject.n_classes} class(es), B={a.batch}, S={a.samples}"
        )
        header = f"{'size':>5} {'backend':>7} {'ms/render':>10} {'±':>6} {'FPS':>7} {'speedup':>8} {'work MB':>8} {'rel err':>9}"
        print(header)

        for size in a.size:
            k = make_k_inv(1020.0, 400.0 / size, 400.0 / size, 0.0, 0.0, size, size)
            k = k.to(device).expand(a.batch, -1, -1)
            args = (subject, k.to(dt), rt_inv_lp, sdd_lp, size, size, a.samples)
            ref_args = (ref_subject, k, rt_inv, sdd, size, size, a.samples)
            ref = (
                GRADS[a.grad](ref_args, "torch", a.soft_labels)
                if a.grad
                else render(*ref_args, backend="torch", soft_labels=a.soft_labels)
            )
            ref = ref.float()
            base_us = None
            for backend in available_backends(device):
                run = (
                    functools.partial(GRADS[a.grad], args, backend, a.soft_labels)
                    if a.grad
                    else functools.partial(render, *args, backend=backend, soft_labels=a.soft_labels)
                )

                err = ((run().float() - ref).abs().max() / ref.abs().max()).item()
                iters = a.cpu_iterations if device == "cpu" else a.iterations
                us, std = time_us(run, device, a.runs if device != "cpu" else 3, iters, 2 if device == "cpu" else 10)
                mem = working_set_mb(run, device)
                base_us = base_us or us
                print(
                    f"{size:>5} {backend:>7} {us / 1e3:>10.2f} {std / 1e3:>6.2f} {1e6 / us:>7.0f} "
                    f"{base_us / us:>7.1f}x {mem:>8.1f} {err:>9.1e}"
                )


if __name__ == "__main__":
    main()
