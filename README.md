# minidynamo

A small bytecode tracer for learning how TorchDynamo captures a graph and caches executable code behind guards. Read the walkthrough in [blog_post.md](blog_post.md).

## Run it

Requires Python **3.10.x**, PyTorch **2.10.0**, and `uv`. Python 3.10 is intentional: this interpreter implements that version's bytecode. The optional Inductor bridge uses private PyTorch APIs.

```bash
uv sync --frozen --group dev
uv run pytest
```

Inductor's CPU tests also need a C++ compiler. CUDA and MPS tests skip when their device is unavailable. PyTorch can warn about missing NumPy; none of these examples require NumPy.

```python
import torch
import minidynamo

@minidynamo.compile
def fn(x, y):
    z = x + y
    w = z * 2
    return w.sum()

x, y = torch.randn(3, 4), torch.randn(3, 4)
torch.testing.assert_close(fn(x, y), fn._original(x, y))
print(len(fn._cache))  # 1
```

The first call records a graph, executes real example operations for metadata, compiles a callable, and runs it. Later calls scan cached guard sets. New input metadata or changed tracing assumptions cause another trace.

## Supported subset

- Python functions with a fixed number of positional arguments, all plain strided `torch.Tensor` objects without custom attributes, returning one tensor.
- Straight-line arithmetic (`+`, `-`, `*`, `/`, `**`), immutable constants, and supported `torch` functions and tensor methods. The explicit allowlists are in `minidynamo/symbolic_interpreter.py`.
- Metadata attributes (`shape`, `dtype`, `device`, `ndim`) and queries (`dim()`, `size()`, `numel()`). Python calls such as `len(x.shape)` operate on constants only.
- Immutable global constants and aliases of supported functions. The interpreter guards the identity of bindings it reads.
- `backend="python"` (default): generate readable Python that replays the graph.
- `backend="jit"`: apply `torch.jit.trace` to the generated function. This deprecated API is included for comparison; its optimizations and performance depend on the environment.

Unsupported features raise: keyword calls, defaults, variadic arguments, closures, loops, branches, arbitrary Python helpers, mutable/global tensors, in-place operations, random operations, tensor-data conversion such as `.item()`, tensor subclasses, and multi-output operations. The toy does not model custom dispatch modes, monkeypatching the `torch.Tensor` class, or concurrent mutation of the function's environment.

Guards check argument count, tensor type/shape/dtype/device/stride/storage offset/view flags/gradient requirements, grad/inference/autocast modes, default dtype, and the current CUDA device, global bindings and accessed `torch` attributes, and the function's code object. They are conservative, and the per-wrapper cache has no eviction or recompilation limit. Low-level `compile_graph` returns an unguarded callable; use the decorator when inputs or captured state may change.

Ordinary autograd works through supported operations in the Python and JIT backends. There is no compiled backward pass, AOTAutograd integration, graph-break fallback, or general Python compatibility.

## Optional Inductor bridge

```python
from minidynamo.inductor import minidynamo_to_inductor

fused = minidynamo_to_inductor(fn._original, x, y)
torch.testing.assert_close(fused(x, y), fn._original(x, y))
```

The bridge translates our graph to FX, lowers it through `make_fx`, then calls the private `compile_fx_inner` API. It bypasses AOTAutograd and its functionalization/decomposition pipeline, so some operations accepted by the tracer (such as device-changing `.to()` calls) can still fail in this backend. It supports **forward inference only**, with inputs that do not require gradients. It checks its tracing assumptions on each call and raises if they change; create a new callable for a new specialization. It is separate from the decorator's two backends.

The parity tests compare complete normalized ATen graphs, nonempty C++ kernel bodies, and results for four small CPU examples against Dynamo's export path. This does not establish equivalence with the full `torch.compile` pipeline. CUDA parity tests compare complete ATen graphs and nonempty Triton source for the same four programs at 32×32 and the longer chain at 2048×2048. Further CUDA tests cover numerical parity across dtypes, gradients in the Python/JIT backends, and non-contiguous inputs.

## Examples and measurements

```bash
uv run python examples/inductor_integration.py
uv run python examples/benchmark.py
uv run python examples/benchmark_mps.py
uv run python examples/benchmark_transformers.py
```

The device examples select CUDA, then MPS, then CPU. The transformer-inspired examples include whole-tensor RMS normalization and softmax; they are not full transformer layers or per-token normalization implementations.

For the article's CUDA experiment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
uv run python examples/benchmark_publication.py --warmup 500 --iterations 1000 --repeats 15 \
  --output benchmarks/h200-publication.json
```

This measures an eleven-operation elementwise chain plus a reduction using float32 inputs in inference mode. It separates raw from guarded callables, warms up each variant, randomizes measurement order, synchronizes trial boundaries, and saves every trial and environment details. CUDA graphs are disabled in both Inductor paths. Times are amortized wall time per call, not isolated kernel latency. The profiler records CUDA event names separately from timing.

The Python backend retains per-operation dispatch and adds guards. It is for learning and inspection. An optimizing backend can change the kernel structure, but no backend is guaranteed to beat eager execution for every shape.

## File map

| File | Role |
| :-- | :-- |
| `minidynamo/__init__.py` | Decorator and cache loop |
| `minidynamo/symbolic_interpreter.py` | Input validation and bytecode handlers |
| `minidynamo/variable_tracker.py` | Symbolic wrappers used during tracing |
| `minidynamo/graph.py` | Graph nodes and naming |
| `minidynamo/compiler.py` | Python source generation and JIT backend |
| `minidynamo/guards.py` | Input and execution-context checks |
| `minidynamo/inductor.py` | Optional inference bridge to private Inductor APIs |
| `tests/` | Behavioral, regression, graph/kernel parity, and CUDA tests |
| `examples/` | Runnable demonstrations and benchmarks |
| `benchmarks/` | Recorded publication measurements |
| `blog_post.md` | Article source; diagrams are in `assets/` |
