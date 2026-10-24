"""
minidynamo: A minimal, educational implementation of TorchDynamo.

TorchDynamo is the tracing engine behind torch.compile(). It intercepts
Python bytecode execution, symbolically interprets it to capture a graph
of tensor operations, hands that graph to a backend, and caches the result
with guards.

This module provides a single entry point: the compile() decorator.

Usage:
    import minidynamo

    @minidynamo.compile
    def fn(x, y):
        z = x + y
        w = z * 2
        return w.sum()

    # First call: traces bytecode → builds graph → compiles → caches → runs
    result = fn(torch.randn(3, 4), torch.randn(3, 4))

    # Second call (same shapes): guards pass → runs cached compiled fn
    result = fn(torch.randn(3, 4), torch.randn(3, 4))

    # Third call (different shapes): guards fail → retraces → recompiles
    result = fn(torch.randn(5, 6), torch.randn(5, 6))

Backends:
    @minidynamo.compile                   # "python" backend (default)
    @minidynamo.compile(backend="jit")    # torch.jit.trace backend
"""

import functools

from .symbolic_interpreter import (
    SymbolicInterpreter,
    validate_function,
    validate_inputs,
)
from .compiler import compile_graph, compile_graph_jit
from .guards import Guard


def compile(fn=None, *, backend="python"):
    """
    Compile a function by tracing its tensor operations into a graph.
    Analogous to torch.compile().

    On its own, graph capture does NOT produce a meaningful speedup. The
    default "python" backend is an educational backend: it turns the graph
    back into a plain Python callable that still dispatches each torch op
    individually. Real speedups come from passing captured graphs to an
    optimizing backend like Inductor, which can fuse and lower those ops.

    Args:
        fn: The function to compile. If None, returns a decorator.
        backend: Compilation backend to use:
            - "python" (default): generates a plain Python function from the
              graph. Captures the graph but does not optimize it.
            - "jit": additionally wraps with torch.jit.trace to produce a
              TorchScript callable; optimization depends on the JIT configuration.

    Limitations:
        minidynamo only supports positional tensor arguments. Constants inside
        the function body (e.g. `x * 2`) are supported, but non-tensor function
        arguments and keyword arguments are intentionally out of scope.

    How this toy works:

    1. TRACE: On first call, run the SymbolicInterpreter over the function's
       bytecodes. This records a Graph and runs real example operations for metadata.

    2. COMPILE: Pass the Graph to the chosen backend, which generates an
       executable callable.

    3. GUARD: Record the input metadata (shapes, dtypes, devices) as guards —
       boolean conditions that must hold for the compiled code to be valid.

    4. CACHE: Store (guards, compiled_fn) so subsequent calls skip steps 1-3.

    5. EXECUTE: On each call, check cached guards. If they pass, call the
       compiled function directly. If they fail, retrace and compile again.
    """
    if backend not in {"python", "jit"}:
        raise ValueError(
            f"Unsupported backend: {backend!r}. "
            "minidynamo supports only 'python' and 'jit'."
        )

    # Support both @compile and @compile(backend="jit") syntax
    if fn is None:
        # Called as @compile(backend="jit") — return a decorator
        return functools.partial(compile, backend=backend)

    validate_function(fn)

    # The cache: a list of (GuardSet, compiled_fn) entries.
    # Real Dynamo stores this on the code object itself as a linked list.
    # We use a simple list — first entry whose guards pass wins.
    cache = []

    @functools.wraps(fn)
    def wrapper(*args):
        validate_inputs(fn, args)

        # Fast path: scan the cache and run the first entry whose guards
        # all pass. This is the path every steady-state call takes.
        for guard_set, compiled_fn in cache:
            if guard_set.check_all(*args):
                # Cache hit — guards pass, use compiled function
                return compiled_fn(*args)

        # Slow path (cache miss): run the full pipeline and cache the result.
        # Step 1: Trace
        interpreter = SymbolicInterpreter(fn, args)
        graph = interpreter.run()

        # Step 2: Compile (dispatch to chosen backend)
        if backend == "jit":
            compiled_fn, _source = compile_graph_jit(graph, args)
        else:
            compiled_fn, _source = compile_graph(graph)

        # Step 3: Guard
        guard_set = interpreter.guards
        guard_set.add(
            Guard(
                lambda *args, code=fn.__code__: fn.__code__ is code,
                "function code is unchanged",
            )
        )

        # Step 4: Cache
        cache.append((guard_set, compiled_fn))

        # Step 5: Execute
        return compiled_fn(*args)

    # Expose internals for debugging / educational purposes
    wrapper._cache = cache
    wrapper._original = fn
    wrapper._backend = backend

    return wrapper
