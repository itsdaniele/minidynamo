"""Optional, inference-only bridge to PyTorch 2.10's private Inductor API.

This bypasses AOTAutograd and the normal torch.compile backend pipeline.
The returned callable checks its assumptions but does not recompile.
"""

import torch
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx

from minidynamo.symbolic_interpreter import SymbolicInterpreter
from minidynamo.graph import Node
from minidynamo.guards import Guard
from minidynamo.symbolic_interpreter import validate_inputs


# ── Step 1: Convert our Graph IR to a real torch.fx.GraphModule ──────


def to_fx_graph_module(graph):
    """
    Convert our minidynamo Graph into a torch.fx.GraphModule.

    Our Graph uses the same conceptual structure as fx.Graph (placeholder,
    call_function, call_method, output nodes), so the translation is direct.
    """
    fx_graph = fx.Graph()

    # Map our Node -> fx.Node so we can resolve references
    node_map = {}

    def convert(value):
        if isinstance(value, Node):
            return node_map[value]
        if isinstance(value, tuple):
            return tuple(convert(v) for v in value)
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    for node in graph.nodes:
        if node.op == "placeholder":
            fx_node = fx_graph.placeholder(node.name)
            node_map[node] = fx_node

        elif node.op == "call_function":
            # Translate args: our Nodes -> fx Nodes, constants stay as-is
            fx_args = convert(node.args)
            fx_kwargs = convert(node.kwargs)
            fx_node = fx_graph.call_function(node.target, fx_args, fx_kwargs)
            fx_node.name = node.name
            node_map[node] = fx_node

        elif node.op == "call_method":
            fx_args = convert(node.args)
            fx_kwargs = convert(node.kwargs)
            fx_node = fx_graph.call_method(node.target, fx_args, fx_kwargs)
            fx_node.name = node.name
            node_map[node] = fx_node

        elif node.op == "output":
            # Inductor expects the output to be a tuple of tensors
            ret_node = node_map[node.args[0]]
            fx_graph.output((ret_node,))

    # Wrap in a GraphModule (an nn.Module whose forward() is the graph)
    fx_graph.lint()
    gm = fx.GraphModule(torch.nn.Module(), fx_graph)
    return gm


# ── Step 2: Lower to ATen ops ───────────────────────────────────────


def lower_to_aten(gm, *example_inputs):
    """
    Lower high-level torch ops to ATen ops using make_fx.

    torch.add -> torch.ops.aten.add.Tensor
    torch.relu -> torch.ops.aten.relu.default
    x.sum()   -> torch.ops.aten.sum.default

    Inductor operates on ATen ops, not Python-level torch ops.
    """
    aten_gm = make_fx(gm)(*example_inputs)
    return aten_gm


# ── Step 3: Compile with Inductor ────────────────────────────────────


def compile_with_inductor(aten_gm, example_inputs):
    """
    Pass the ATen-lowered graph to Inductor for kernel fusion.

    compile_fx_inner is the internal entry point that takes:
      - An fx.GraphModule with ATen ops
      - Example inputs for shape inference

    It returns a CompiledFxGraph whose __call__ takes a SINGLE list of
    tensors (not individual args). This is Inductor's calling convention.
    """
    from torch._inductor.compile_fx import compile_fx_inner

    compiled = compile_fx_inner(aten_gm, list(example_inputs))
    return compiled


# ── Full pipeline ────────────────────────────────────────────────────


def minidynamo_to_inductor(fn, *example_inputs, verbose=False):
    """
    End-to-end: user function -> minidynamo trace -> Inductor compiled.

    Returns a callable that runs the Inductor-fused version.
    """
    # 1. Trace with our symbolic interpreter
    interpreter = SymbolicInterpreter(fn, example_inputs)
    if any(arg.requires_grad for arg in example_inputs):
        raise NotImplementedError(
            "The direct Inductor example is inference-only; detach the inputs."
        )
    graph = interpreter.run()
    guards = interpreter.guards
    guards.add(
        Guard(
            lambda *args, code=fn.__code__: fn.__code__ is code,
            "function code is unchanged",
        )
    )
    if verbose:
        print("Our captured graph:")
        print(graph)
        print()

    # 2. Convert to fx.GraphModule
    gm = to_fx_graph_module(graph)
    if verbose:
        print("FX GraphModule:")
        print(gm.graph)

    # 3. Lower to ATen ops
    aten_gm = lower_to_aten(gm, *example_inputs)
    if verbose:
        print("ATen-lowered graph:")
        print(aten_gm.graph)
        print()

    # 4. Compile with Inductor
    compiled = compile_with_inductor(aten_gm, example_inputs)

    # 5. Wrap to match the original function's calling convention
    # Inductor's compiled function takes a single list of tensors
    # and returns a tuple of outputs. We unwrap both.
    def wrapper(*args):
        validate_inputs(fn, args)
        if not guards.check_all(*args):
            raise ValueError(
                "Inductor example inputs or tracing assumptions changed; compile a new callable."
            )
        results = compiled(list(args))
        return results[0]

    return wrapper
