"""Reproducible CUDA timings for the article; writes raw trials and provenance.

Times synchronized blocks of calls, including host dispatch and wrapper guards.
This measures amortized wall time, not isolated kernel latency. Compilation is
excluded. CUDA graphs are disabled in both Inductor paths to compare codegen.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import torch
import torch._inductor.config as inductor_config

import minidynamo
from minidynamo.compiler import compile_graph, compile_graph_jit
from minidynamo.inductor import minidynamo_to_inductor
from minidynamo.symbolic_interpreter import SymbolicInterpreter


def many_ops(x, y):
    """Eleven elementwise operations followed by a sum."""
    a = x + y
    b = a * 2
    c = b - x
    d = c * y
    e = torch.relu(d)
    f = e + x
    g = f * 3
    h = torch.sigmoid(g)
    i = h - y
    j = i * x
    k = torch.tanh(j)
    return k.sum()


def time_block(fn, args, iterations):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def kernel_events(fn, args):
    from torch.profiler import ProfilerActivity, profile, record_function

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        # Warm up the activity collector too: its first invocation can miss
        # initial CUDA events even when the callable itself is already warm.
        fn(*args)
        torch.cuda.synchronize()
        for i in range(2):
            with record_function(f"measured_call_{i}"):
                fn(*args)
                torch.cuda.synchronize()

    def activities(event):
        # Follow CPU/CUDA correlations to exclude the collector warmup and
        # GPU annotation ranges. Memcpy/memset activities remain identifiable.
        return [kernel.name for kernel in event.kernels] + [
            name for child in event.cpu_children for name in activities(child)
        ]

    samples = [
        activities(event)
        for event in prof.events()
        if event.device_type.name == "CPU"
        and event.name in {"measured_call_0", "measured_call_1"}
    ]
    if len(samples) != 2 or not samples[0] or samples[0] != samples[1]:
        raise RuntimeError("CUDA profiler samples are missing or inconsistent.")
    return samples[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 256, 2048])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = random.Random(0)
    props = torch.cuda.get_device_properties(0)
    result = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": props.name,
        "gpu_memory_bytes": props.total_memory,
        "capability": list(torch.cuda.get_device_capability()),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__).relative_to(Path.cwd()),
                *sorted(Path("minidynamo").glob("*.py")),
            ]
        },
        "dtype": "float32",
        "inference_mode": True,
        "cuda_graphs": False,
        "torch_threads": torch.get_num_threads(),
        "seed": 0,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "profiler_warmup_calls": 1,
        "profiler_checked_calls": 2,
        "metric": "median amortized wall-clock microseconds per call; synchronized trial boundaries; compilation excluded",
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), inductor_config.patch({"triton.cudagraphs": False}):
        for size in args.sizes:
            inputs = (
                torch.randn(size, size, device="cuda"),
                torch.randn(size, size, device="cuda"),
            )
            graph = SymbolicInterpreter(many_ops, inputs).run()
            python_fn, _ = compile_graph(graph)
            jit_fn, _ = compile_graph_jit(graph, inputs)
            torch._dynamo.reset()
            variants = {
                "eager": many_ops,
                "python_raw": python_fn,
                "python_guarded": minidynamo.compile(many_ops),
                "jit_raw": jit_fn,
                "jit_guarded": minidynamo.compile(many_ops, backend="jit"),
                "mini_inductor_guarded": minidynamo_to_inductor(many_ops, *inputs),
                "torch_compile": torch.compile(many_ops, fullgraph=True, dynamic=False),
            }
            expected = many_ops(*inputs)
            errors = {}
            for name, fn in variants.items():
                actual = fn(*inputs)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
                # Also check new tensor contents with the same specialization.
                fresh = tuple(x * 0.75 for x in inputs)
                torch.testing.assert_close(
                    fn(*fresh), many_ops(*fresh), rtol=1e-4, atol=1e-3
                )
                errors[name] = float((actual - expected).abs().item())
                for _ in range(args.warmup):
                    fn(*inputs)
            trials = {name: [] for name in variants}
            for _ in range(args.repeats):
                order = list(variants)
                rng.shuffle(order)
                for name in order:
                    trials[name].append(
                        time_block(variants[name], inputs, args.iterations)
                    )
            for name, fn in variants.items():
                samples = trials[name]
                row = {
                    "shape": [size, size],
                    "backend": name,
                    "median_us": statistics.median(samples),
                    "min_us": min(samples),
                    "max_us": max(samples),
                    "trials_us": samples,
                    "max_abs_error": errors[name],
                }
                try:
                    row["cuda_events"] = kernel_events(fn, inputs)
                    row["cuda_event_count"] = len(row["cuda_events"])
                    # CUDA activity also includes memsets and Inductor's
                    # annotation ranges; those are not kernel launches.
                    row["kernel_names"] = [
                        name
                        for name in row["cuda_events"]
                        if not name.startswith(("Memset", "Memcpy", "##"))
                    ]
                    row["kernel_count"] = len(row["kernel_names"])
                except RuntimeError as exc:
                    row["profiler_error"] = str(exc)
                result["rows"].append(row)
                print(
                    f"{size:4} {name:24} {row['median_us']:9.2f} us  CUDA events={row.get('cuda_event_count', 'unavailable')}",
                    flush=True,
                )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
