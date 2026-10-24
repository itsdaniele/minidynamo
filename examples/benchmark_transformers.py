"""
Transformer Ops Fusion Benchmark: minidynamo + Inductor vs torch.compile

Compares three execution modes on GPU (CUDA or MPS):
  - Eager:         plain PyTorch, no compilation
  - mini+Inductor: our minidynamo captures the graph, Inductor fuses kernels
  - torch.compile: real TorchDynamo + Inductor

Patterns benchmarked: GELU, SiLU, RMSNorm, Softmax, Residual+GELU.
"""

import time
import torch
import torch._dynamo as dynamo

from minidynamo.inductor import minidynamo_to_inductor as compile_fn

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def bench(fn, *args, warmup=50, iters=1000):
    for _ in range(warmup):
        fn(*args)
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    sync()
    return (time.perf_counter() - start) / iters


# ── Transformer patterns ─────────────────────────────────────────────


def gelu(x):
    """GELU activation (approximate, GPT-2/BERT). 8 elementwise ops."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def silu(x):
    """SiLU / Swish (LLaMA, Mistral). 2 elementwise ops."""
    return x * torch.sigmoid(x)


def rms_norm(x):
    """RMS normalization over the whole tensor (not per-token RMSNorm)."""
    rms = torch.sqrt((x * x).mean() + 1e-6)
    return x / rms


def softmax(x):
    """Numerically stable softmax over all elements, not the last dimension."""
    x_max = x.max()
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum()


def residual_gelu(x, residual):
    """Residual + GELU. 9 ops: add + full GELU chain."""
    h = x + residual
    return 0.5 * h * (1.0 + torch.tanh(0.7978845608 * (h + 0.044715 * h**3)))


# ── Benchmark runner ─────────────────────────────────────────────────

PATTERNS = [
    ("GELU (8 ops)", gelu, 1),
    ("SiLU (2 ops)", silu, 1),
    ("Whole-tensor RMS norm", rms_norm, 1),
    ("Whole-tensor softmax", softmax, 1),
    ("Residual+GELU (9 ops)", residual_gelu, 2),
]

if __name__ == "__main__":
    SEQ, D = 512, 768

    print("=" * 78)
    print(f"Transformer Ops Fusion: Eager vs minidynamo vs torch.compile [{DEVICE}]")
    print(f"Tensor size: ({SEQ}, {D}) -- GPT-2 small dimensions")
    print("All runs use torch.inference_mode() (no autograd overhead)")
    print("=" * 78)
    print()

    with torch.inference_mode():
        x = torch.randn(SEQ, D, device=DEVICE)
        residual = torch.randn(SEQ, D, device=DEVICE)

        # Pre-build torch.compile versions (to avoid measuring compilation time)
        dynamo.reset()
        tc_fns = {}
        for name, fn, nargs in PATTERNS:
            args = (x,) if nargs == 1 else (x, residual)
            tc = torch.compile(fn, backend="inductor")
            tc(*args)  # warmup / compile
            sync()
            tc_fns[name] = tc

        # Header
        print(
            f"  {'Pattern':<26} {'Eager':>10} {'mini+Ind':>10} "
            f"{'torch.compile':>14} {'mini spd':>9} {'tc spd':>9}"
        )
        print(f"  {'-' * 26} {'-' * 10} {'-' * 10} {'-' * 14} {'-' * 9} {'-' * 9}")

        for name, fn, nargs in PATTERNS:
            args = (x,) if nargs == 1 else (x, residual)

            # Compile with our pipeline
            mini_fn = compile_fn(fn, *args)

            # Verify correctness
            expected = fn(*args)
            mini_result = mini_fn(*args)
            tc_result = tc_fns[name](*args)
            assert torch.allclose(expected, mini_result, atol=1e-3), (
                f"{name}: mini mismatch!"
            )
            assert torch.allclose(expected, tc_result, atol=1e-3), (
                f"{name}: tc mismatch!"
            )

            # Benchmark
            t_eager = bench(fn, *args)
            t_mini = bench(mini_fn, *args)
            t_tc = bench(tc_fns[name], *args)

            sp_mini = t_eager / t_mini
            sp_tc = t_eager / t_tc

            print(
                f"  {name:<26} {t_eager * 1e6:>8.1f}us {t_mini * 1e6:>8.1f}us "
                f"{t_tc * 1e6:>12.1f}us {sp_mini:>8.2f}x {sp_tc:>8.2f}x"
            )

    print()
    print("=" * 78)
    print("ANALYSIS")
    print("=" * 78)
    print("""
  mini+Inductor and torch.compile use the same Inductor backend, and the
  parity tests check that minidynamo can feed equivalent graphs into that
  narrow path. When these rows speed up, the win comes from Inductor's
  lowering and fusion work, not from our graph capture.

  The two paths have different guards, runtime wrappers, and preprocessing.
  Our bridge bypasses AOTAutograd and compiles forward inference only.
  These timings alone cannot attribute the difference to any one layer.
  Small kernels, reductions, and backend-specific implementations can be
  slower than eager. Use the table above as the result for this machine.
""")
