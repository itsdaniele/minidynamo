"""
Benchmark on GPU (CUDA or MPS): minidynamo backends vs torch.compile

This benchmark compares execution modes on GPU:

  1. Eager PyTorch on GPU — baseline. Each operation is a separate GPU
     kernel dispatched from Python.

  2. minidynamo "python" backend — generated Python from the captured
     graph, but still separate GPU kernels through the normal dispatcher.
     This is a correctness/debugging backend, not a performance backend.

  3. minidynamo "jit" backend — torch.jit.trace produces a TorchScript
     function that executes in C++. It can reduce Python/C++ boundary
     overhead and may apply JIT optimizations; measure rather than assuming
     a fixed kernel count.

  4. torch.compile + Inductor — generates FUSED GPU kernels (Triton on
     CUDA, Metal on MPS where supported). Instead of N separate GPU
     kernels, elementwise ops may become ONE kernel that reads the tensor
     once, computes everything, and writes once. This can be the real
     speedup path, but the result is backend- and shape-dependent.
"""

import time
import warnings
import torch
import torch._dynamo as dynamo

# Suppress JIT deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.jit")

from minidynamo.symbolic_interpreter import SymbolicInterpreter
from minidynamo.compiler import compile_graph, compile_graph_jit


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def sync():
    """Synchronize the GPU to get accurate wall-clock times."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def benchmark_gpu(fn, *args, warmup=50, iterations=2000):
    """
    Benchmark a function with GPU synchronization.

    On GPU, ops are asynchronous — fn() returns before the GPU finishes.
    We must synchronize to get accurate wall-clock times.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args)
    sync()

    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    sync()
    elapsed = time.perf_counter() - start

    return elapsed / iterations


# ── Test function: 11 chained elementwise operations ──────────────────
# This is a useful kernel-fusion case: many cheap ops that are memory-bound
# when each op reads and writes the full tensor separately.


def many_ops(x, y):
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


def run_benchmarks():
    print("=" * 72)
    print(f"minidynamo GPU Benchmark (device: {DEVICE})")
    print("=" * 72)
    print()
    print("Function: many_ops() — 11 chained elementwise operations + sum")
    print("This is a useful fusion experiment, but backend results can vary.")
    print()

    # ── Prepare all the compiled variants ──────────────────────────────

    sizes = [(32, 32), (128, 128), (512, 512), (2048, 2048)]

    for H, W in sizes:
        x = torch.randn(H, W, device=DEVICE)
        y = torch.randn(H, W, device=DEVICE)

        # 1. Raw compiled fn (no guard overhead)
        graph = SymbolicInterpreter(many_ops, (x, y)).run()
        python_fn, _ = compile_graph(graph)
        jit_fn, _ = compile_graph_jit(graph, (x, y))

        # 2. Inductor (real torch.compile)
        # Reset between shapes so Dynamo does not generalize this same Python
        # function across previous benchmark sizes.
        try:
            dynamo.reset()
            inductor_fn = torch.compile(many_ops, backend="inductor")
            # Warmup inductor (first call triggers compilation)
            inductor_fn(x, y)
            sync()
            has_inductor = True
        except Exception:
            has_inductor = False

        # ── Benchmark ──────────────────────────────────────────────────

        t_eager = benchmark_gpu(many_ops, x, y)
        t_python = benchmark_gpu(python_fn, x, y)
        t_jit = benchmark_gpu(jit_fn, x, y)

        # ── Verify correctness ─────────────────────────────────────────

        expected = many_ops(x, y)
        assert torch.allclose(expected.cpu(), python_fn(x, y).cpu(), atol=1e-5)
        assert torch.allclose(expected.cpu(), jit_fn(x, y).cpu(), atol=1e-5)

        # ── Print results ──────────────────────────────────────────────

        print(f"  Tensor size: {H}x{W} ({H * W:,} elements)")
        print(f"  {'Eager':<28} {t_eager * 1e6:>8.1f} us  (baseline)")
        print(
            f"  {'minidynamo (python)':<28} {t_python * 1e6:>8.1f} us  ({t_eager / t_python:.2f}x)"
        )
        print(
            f"  {'minidynamo (jit)':<28} {t_jit * 1e6:>8.1f} us  ({t_eager / t_jit:.2f}x)"
        )

        if has_inductor:
            t_inductor = benchmark_gpu(inductor_fn, x, y)
            assert torch.allclose(expected.cpu(), inductor_fn(x, y).cpu(), atol=1e-4)
            print(
                f"  {'torch.compile (inductor)':<28} {t_inductor * 1e6:>8.1f} us  ({t_eager / t_inductor:.2f}x)"
            )

        print()

    # ── Explanation ────────────────────────────────────────────────────

    print("=" * 72)
    print("WHAT EACH BACKEND CAN AND CANNOT CHANGE")
    print("=" * 72)
    print("""
  Eager: Python interpreter runs between every GPU op. Each op is a
  separate GPU kernel: read tensor → compute → write tensor.

  minidynamo (python): Same GPU kernels and the same dispatcher path.
  The generated Python backend is useful for inspecting graph capture,
  but it is not expected to produce a meaningful speedup.

  minidynamo (jit): TorchScript executes in C++ and can reduce Python/C++
  boundary overhead. Its own optimization passes may change the graph.

  torch.compile (inductor): Can fuse elementwise ops into a SINGLE GPU
  kernel (Triton on CUDA, Metal on MPS where supported). One read, all
  operations computed in GPU registers, one write. When this wins, the
  win comes from changing the kernel structure, not from minidynamo's
  graph capture.

  Larger tensors, reductions, device backend maturity, and wrapper
  overhead can all change the result. Treat the table above as the
  measurement for this machine, not a portable performance promise.
""")


if __name__ == "__main__":
    run_benchmarks()
