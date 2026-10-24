"""
Compiler Backend — turns a captured Graph into a callable.

In real TorchDynamo, the backend is pluggable: you can use TorchInductor
(generates Triton GPU kernels and C++ CPU code), or a custom backend.
torch.compile(fn, backend="inductor") uses Inductor by default.

Our backend is simpler: it takes the captured graph and produces a plain
Python callable that replays the same torch operations one at a time. This
does NOT provide a meaningful speedup on its own; it is an educational
backend whose main value is making the captured graph executable and easy
to inspect.

The real value of graph capture is enabling a backend like Inductor to fuse
multiple operations into a single GPU kernel, which eliminates memory
round-trips and provides the actual speedup.
"""

import math

import torch

from .graph import Node


def compile_graph(graph):
    """
    Compile a Graph into a plain Python callable.

    Strategy: generate Python source code for a function that directly
    calls the recorded torch operations, then exec() it.

    For this graph:
        x = placeholder
        y = placeholder
        add_0 = torch.add(x, y)
        mul_0 = torch.mul(add_0, 2)
        sum_0 = mul_0.sum()
        return sum_0

    We generate:
        def compiled_fn(x, y):
            add_0 = __fn_add_0(x, y)
            mul_0 = __fn_mul_0(add_0, 2)
            sum_0 = mul_0.sum()
            return sum_0

    Where __fn_add_0, __fn_mul_0 are torch.add, torch.mul stored in the
    function's globals. Each generated line still calls the normal PyTorch operation;
    there is no kernel fusion or dispatcher bypass in this backend.

    Returns:
        (compiled_fn, source_code) — the callable and its source for debugging
    """
    # 1. Build the function signature from placeholder names
    param_names = [n.name for n in graph.inputs]
    signature = ", ".join(param_names)

    # 2. Generate the function body — one line per operation node
    body_lines = []
    closure_vars = {}  # name -> Python object, becomes the exec namespace
    reserved = {n.name for n in graph.nodes} | {"compiled_fn"}

    def bind(value, preferred):
        # Constants such as dtype/device and non-finite floats are not
        # necessarily valid Python literals. Keep those objects in globals.
        name = preferred
        while name in reserved:
            name += "_"
        reserved.add(name)
        closure_vars[name] = value
        return name

    def format_arg(arg):
        if isinstance(arg, Node):
            return arg.name
        if isinstance(arg, tuple):
            return "(" + ", ".join(format_arg(a) for a in arg) + ",)" if arg else "()"
        if isinstance(arg, list):
            return "[" + ", ".join(format_arg(a) for a in arg) + "]"
        if isinstance(arg, dict):
            return (
                "{"
                + ", ".join(f"{format_arg(k)}: {format_arg(v)}" for k, v in arg.items())
                + "}"
            )
        if type(arg) in (int, bool, str, type(None)):
            return repr(arg)
        if isinstance(arg, float) and math.isfinite(arg):
            return repr(arg)
        return bind(arg, "__constant")

    def format_args(args, kwargs):
        return ", ".join(
            [format_arg(a) for a in args]
            + [f"{key}={format_arg(value)}" for key, value in kwargs.items()]
        )

    for node in graph.nodes:
        if node.op == "placeholder":
            # Inputs come from function parameters — nothing to generate
            continue

        elif node.op == "call_function":
            # Generate: add_0 = __fn_add_0(x, y)
            # Store the actual function in the exec namespace as __fn_add_0
            closure_key = bind(node.target, f"__fn_{node.name}")

            args_str = format_args(node.args, node.kwargs)
            body_lines.append(f"    {node.name} = {closure_key}({args_str})")

        elif node.op == "call_method":
            # Generate: sum_0 = mul_0.sum()
            self_name = format_arg(node.args[0])
            rest_args = node.args[1:]
            args_str = format_args(rest_args, node.kwargs)
            body_lines.append(
                f"    {node.name} = {self_name}.{node.target}({args_str})"
            )

        elif node.op == "output":
            ret_name = format_arg(node.args[0])
            body_lines.append(f"    return {ret_name}")

        else:
            raise NotImplementedError(f"Unsupported graph operation: {node.op}")

    # 3. Assemble the full source
    body = "\n".join(body_lines)
    source = f"def compiled_fn({signature}):\n{body}"

    # 4. Compile and exec
    code = compile(source, "<minidynamo-compiled>", "exec")
    exec(code, closure_vars)
    compiled_fn = closure_vars["compiled_fn"]

    # exec() doesn't set __module__, which torch.jit.trace needs
    compiled_fn.__module__ = "minidynamo.compiled"

    return compiled_fn, source


def compile_graph_jit(graph, example_inputs):
    """
    JIT-trace backend: graph → Python function → TorchScript via torch.jit.trace.

    TorchScript executes the recorded graph without returning to Python
    between operations. It has its own optimization passes; neither fusion
    nor a speedup is guaranteed by this wrapper. Measure the resulting code.

    Returns:
        (traced_fn, source_code) — the TorchScript callable and the
        intermediate Python source for debugging
    """
    # Step 1: Generate the Python function (same as the "python" backend)
    compiled_fn, source = compile_graph(graph)

    # Step 2: Trace it with example inputs to get a TorchScript function.
    # torch.jit.trace runs the function with real inputs (including checking runs), records the
    # operations as a TorchScript IR graph, and returns an optimized callable.
    traced_fn = torch.jit.trace(compiled_fn, example_inputs)

    return traced_fn, source


def _arg_to_str(arg):
    """Convert a graph arg to its source code representation."""
    if isinstance(arg, Node):
        return arg.name
    return repr(arg)


def _format_call_args(args):
    """Format args tuple as comma-separated source code."""
    return ", ".join(_arg_to_str(a) for a in args)
