"""
Benchmark: graph capture alone vs. an optimizing backend.

minidynamo's default Python backend is a correctness/debugging backend: it
turns the captured graph back into a plain Python callable that still dispatches
each torch op individually. It should not be expected to speed up real tensor
programs. The meaningful speedup in torch.compile comes from handing captured
graphs to an optimizing backend like Inductor, which can fuse ops and reduce
memory traffic / launch overhead.
"""

import time
import torch
import torch._dynamo as dynamo
import minidynamo
from minidynamo.symbolic_interpreter import SymbolicInterpreter
from minidynamo.compiler import compile_graph


def benchmark(fn, *args, warmup=50, iterations=5000):
    """Time a function over many iterations. Returns seconds per call."""
    for _ in range(warmup):
        fn(*args)
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    elapsed = time.perf_counter() - start
    return elapsed / iterations


def trace_and_compile(fn, *args):
    """Get JUST the compiled function (no guard-checking wrapper)."""
    graph = SymbolicInterpreter(fn, args).run()
    compiled_fn, source = compile_graph(graph)
    return compiled_fn, source


# ══════════════════════════════════════════════════════════════════════
# Test function: many chained operations
# ══════════════════════════════════════════════════════════════════════


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


def simple_fn(x, y):
    return (x + y) * 2


if __name__ == "__main__":
    print("=" * 70)
    print("minidynamo Benchmark: Where Does Speedup Come From?")
    print("=" * 70)

    # ── Show the generated code ────────────────────────────────────────
    print()
    print("STEP 1: Trace and compile many_ops()")
    print("-" * 50)

    x = torch.randn(100, 100)
    y = torch.randn(100, 100)

    graph = SymbolicInterpreter(many_ops, (x, y)).run()
    print("Captured graph:")
    print(graph)
    print()

    compiled_fn, source = compile_graph(graph)
    print("Generated source:")
    print(source)
    print()

    # Verify correctness
    assert torch.allclose(many_ops(x, y), compiled_fn(x, y))
    print("Correctness verified!")

    # ── Benchmark 1: generated Python function directly (no guards) ─────
    # This is a sanity check for graph capture/codegen, not a speedup path.
    print()
    print("=" * 70)
    print("BENCHMARK 1: Raw compiled function vs eager (NO guard overhead)")
    print("This generated function still calls each torch op individually.")
    print("=" * 70)

    for label, size in [
        ("tiny (4)", 4),
        ("small (32)", 32),
        ("medium (256)", 256),
        ("large (1024)", 1024),
    ]:
        x = torch.randn(size, size)
        y = torch.randn(size, size)
        compiled_fn, _ = trace_and_compile(many_ops, x, y)

        t_eager = benchmark(many_ops, x, y)
        t_compiled = benchmark(compiled_fn, x, y)
        speedup = t_eager / t_compiled
        print(
            f"  {label:<20}  eager={t_eager * 1e6:>8.1f} us  compiled={t_compiled * 1e6:>8.1f} us  {speedup:.2f}x"
        )

    # ── Benchmark 2: with guard overhead (full decorator) ──────────────
    print()
    print("=" * 70)
    print("BENCHMARK 2: Full minidynamo.compile (WITH guard overhead)")
    print("Guards add overhead. Real Dynamo minimizes this with C guards.")
    print("=" * 70)

    many_ops_decorated = minidynamo.compile(many_ops)

    for label, size in [
        ("tiny (4)", 4),
        ("small (32)", 32),
        ("medium (256)", 256),
        ("large (1024)", 1024),
    ]:
        x = torch.randn(size, size)
        y = torch.randn(size, size)

        t_eager = benchmark(many_ops, x, y)
        t_decorated = benchmark(many_ops_decorated, x, y)
        speedup = t_eager / t_decorated
        print(
            f"  {label:<20}  eager={t_eager * 1e6:>8.1f} us  compiled={t_decorated * 1e6:>8.1f} us  {speedup:.2f}x"
        )

    # ── Benchmark 3: Guard overhead around the educational backend ──────
    print()
    print("=" * 70)
    print("BENCHMARK 3: Guard overhead (256x256 tensors)")
    print("=" * 70)

    x = torch.randn(256, 256)
    y = torch.randn(256, 256)
    compiled_fn, _ = trace_and_compile(many_ops, x, y)
    many_ops_decorated = minidynamo.compile(many_ops)
    many_ops_decorated(x, y)  # populate cache

    t_eager = benchmark(many_ops, x, y)
    t_raw_compiled = benchmark(compiled_fn, x, y)
    t_full_decorator = benchmark(many_ops_decorated, x, y)

    guard_overhead = t_full_decorator - t_raw_compiled
    raw_delta = t_eager - t_raw_compiled

    print(f"  Eager (original fn):          {t_eager * 1e6:>8.1f} us")
    print(f"  Raw compiled (no guards):     {t_raw_compiled * 1e6:>8.1f} us")
    print(f"  Full decorator (with guards): {t_full_decorator * 1e6:>8.1f} us")
    print()
    print(f"  Raw generated fn delta:       {raw_delta * 1e6:>+8.1f} us")
    print(f"  Guard checking added:         {guard_overhead * 1e6:>+8.1f} us")
    print(
        f"  Net effect of decorator:      {(t_eager - t_full_decorator) * 1e6:>+8.1f} us"
    )

    # ── Benchmark 4: vs torch.compile ──────────────────────────────────
    print()
    print("=" * 70)
    print("BENCHMARK 4: vs torch.compile (uses Inductor with kernel fusion)")
    print("=" * 70)

    for label, size in [
        ("small (32)", 32),
        ("medium (256)", 256),
        ("large (1024)", 1024),
    ]:
        x = torch.randn(size, size)
        y = torch.randn(size, size)
        compiled_fn, _ = trace_and_compile(many_ops, x, y)

        # Compile per size so cache state from earlier benchmark shapes does
        # not affect the steady-state timing for this shape.
        try:
            dynamo.reset()
            many_ops_torch_inductor = torch.compile(many_ops, backend="inductor")
            many_ops_torch_inductor(x, y)
            has_inductor = True
        except Exception:
            has_inductor = False

        t_eager = benchmark(many_ops, x, y)
        t_ours = benchmark(compiled_fn, x, y)

        line = f"  {label:<20}  eager={t_eager * 1e6:>8.1f}  ours={t_ours * 1e6:>8.1f} ({t_eager / t_ours:.2f}x)"

        if has_inductor:
            t_inductor = benchmark(
                many_ops_torch_inductor, x, y, warmup=5, iterations=2000
            )
            line += f"  inductor={t_inductor * 1e6:>8.1f} ({t_eager / t_inductor:.2f}x)"

        print(line + "  (us)")

    print()
    print("=" * 70)
    print("TAKEAWAY:")
    print("  minidynamo's Python backend is a graph-capture sanity check,")
    print("  not a performance backend. Real torch.compile speedups come")
    print("  from Inductor fusing/lowering the captured graph.")
    print("=" * 70)
