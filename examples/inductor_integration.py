"""Inspect minidynamo -> FX -> ATen -> Inductor, then compare CPU timings.

The optional bridge uses a private API and supports forward inference only.
"""

import time
import torch
import torch._dynamo as dynamo

from minidynamo.inductor import minidynamo_to_inductor


# ── Demo ─────────────────────────────────────────────────────────────


def many_ops(x, y):
    """11 chained elementwise ops -- useful for fusion experiments."""
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


def bench(fn, *args, warmup=50, iters=3000):
    for _ in range(warmup):
        fn(*args)
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - start) / iters


if __name__ == "__main__":
    print("=" * 70)
    print("minidynamo -> Inductor Integration")
    print("=" * 70)
    print()

    # ── Show the full pipeline (verbose) for one example ─────────────

    x = torch.randn(256, 256)
    y = torch.randn(256, 256)

    print("PIPELINE WALKTHROUGH (256x256 tensors)")
    print("-" * 70)
    print()
    inductor_fn = minidynamo_to_inductor(many_ops, x, y, verbose=True)

    # Verify correctness
    expected = many_ops(x, y)
    result = inductor_fn(x, y)
    print(f"Eager result:    {expected.item():.6f}")
    print(f"Inductor result: {result.item():.6f}")
    assert torch.allclose(expected, result, atol=1e-4), "Results don't match!"
    print("Correctness verified!")
    print()

    # ── Benchmark: minidynamo+Inductor vs torch.compile ────────────

    print("=" * 70)
    print("BENCHMARK: minidynamo+Inductor vs torch.compile (CPU)")
    print("=" * 70)
    print()
    print("Function: many_ops() -- 11 chained elementwise ops + sum")
    print()
    print("Both pipelines use the same Inductor backend for kernel fusion.")
    print("Graph capture differs, and our bridge bypasses AOTAutograd:")
    print("  - minidynamo: our bytecode interpreter (SymbolicInterpreter)")
    print("  - torch.compile: real TorchDynamo (PEP 523 frame evaluation)")
    print()

    header = (
        f"  {'Size':<12}  {'Eager':>10}  {'mini+Inductor':>14}  "
        f"{'torch.compile':>14}  {'mini speedup':>12}  {'tc speedup':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, size in [("64x64", 64), ("256x256", 256), ("1024x1024", 1024)]:
        x = torch.randn(size, size)
        y = torch.randn(size, size)

        ind_fn = minidynamo_to_inductor(many_ops, x, y)
        # Compile torch.compile per size so prior benchmark shapes do not
        # trigger Dynamo's dynamic-shape generalization path.
        dynamo.reset()
        torch_compiled = torch.compile(many_ops, backend="inductor")
        expected = many_ops(x, y)
        torch.testing.assert_close(ind_fn(x, y), expected, rtol=1e-4, atol=1e-3)
        torch.testing.assert_close(torch_compiled(x, y), expected, rtol=1e-4, atol=1e-3)

        t_eager = bench(many_ops, x, y)
        t_mini = bench(ind_fn, x, y)
        t_tc = bench(torch_compiled, x, y)

        print(
            f"  {label:<12}  {t_eager * 1e6:>8.1f} us  {t_mini * 1e6:>12.1f} us  "
            f"{t_tc * 1e6:>12.1f} us  {t_eager / t_mini:>10.2f}x  "
            f"{t_eager / t_tc:>10.2f}x"
        )

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
  Dynamo captures the graph. Inductor optimizes it. The parity tests compare
  minidynamo and Dynamo export through a shared lowering path on a small
  set of examples. They check graph wiring, generated kernels, and results.

  Sharing a backend does not guarantee identical timings: torch.compile
  also has different graph preparation, guards, and runtime wrappers.
  Our direct bridge is limited to forward inference and bypasses AOTAutograd.
""")
