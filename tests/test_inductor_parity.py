"""Compare four programs against Dynamo export and a shared Inductor path.

Both captured graphs are lowered with make_fx and compiled via the private
compile_fx_inner API. This does not compare the full torch.compile pipeline.
Tests check operation wiring, constants, nonempty C++/Triton kernels, and results.
"""

import ast
import re
import torch
import torch._dynamo as dynamo
import pytest
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import compile_fx_inner

from minidynamo.symbolic_interpreter import SymbolicInterpreter
from minidynamo.inductor import to_fx_graph_module, minidynamo_to_inductor

import torch.fx as fx


# ── Helpers ───────────────────────────────────────────────────────────


def get_aten_ops_mini(fn, *example_inputs):
    """Get ATen op list from the minidynamo pipeline."""
    graph = SymbolicInterpreter(fn, example_inputs).run()
    gm = to_fx_graph_module(graph)
    aten_gm = make_fx(gm)(*example_inputs)
    return aten_gm


def get_aten_ops_dynamo(fn, *example_inputs):
    """Get ATen op list from the real TorchDynamo pipeline."""
    dynamo.reset()
    exported_gm, _ = dynamo.export(fn)(*example_inputs)
    aten_gm = make_fx(exported_gm)(*example_inputs)
    output = next(iter(reversed(aten_gm.graph.nodes)))
    if not isinstance(output.args[0], (tuple, list)):
        output.args = ((output.args[0],),)
        aten_gm.recompile()
    return aten_gm


def extract_op_list(aten_gm):
    """Extract the ordered list of ATen operations from a graph module."""
    ops = []
    for node in aten_gm.graph.nodes:
        if node.op == "call_function":
            ops.append(str(node.target))
        elif node.op == "call_method":
            ops.append(f"method:{node.target}")
    return ops


def compile_and_get_source(aten_gm, example_inputs):
    """Compile an ATen graph with Inductor and return the source code."""
    # Ensure output is a tuple (Inductor requirement)
    output_node = list(aten_gm.graph.nodes)[-1]
    if output_node.op == "output" and not isinstance(output_node.args[0], tuple):
        output_node.args = ((output_node.args[0],),)
        aten_gm.graph.lint()
        aten_gm.recompile()

    compiled = compile_fx_inner(aten_gm, list(example_inputs))
    return compiled.source_code


def extract_kernel_bodies(source_code):
    """Extract just the C++ kernel function bodies from Inductor source.

    Strips variable names and addresses that differ between compilations,
    keeping only the computational structure.
    """
    # Find all C++ kernel code blocks (between r''' and ''')
    pattern = r"r'''(.*?)'''"
    kernels = re.findall(pattern, source_code, re.DOTALL)
    return kernels


def normalize_kernel(kernel_source):
    """Normalize a kernel source for comparison.

    Removes things that vary between compilations but don't affect
    the actual computation (pointer names, specific sizes in comments, etc.)
    """
    # Remove the #include line
    lines = kernel_source.strip().split("\n")
    lines = [l for l in lines if not l.strip().startswith("#include")]
    # Remove empty lines at start/end
    text = "\n".join(lines).strip()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text


# ── Test functions ────────────────────────────────────────────────────


def simple_add(x, y):
    return (x + y).sum()


def chain_ops(x, y):
    a = x + y
    b = a * 2
    c = b - x
    return c.sum()


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


def relu_sigmoid(x, y):
    return torch.sigmoid(torch.relu(x + y)).sum()


# ── ATen graph parity tests ──────────────────────────────────────────


@pytest.mark.parametrize(
    "fn,name",
    [
        (simple_add, "simple_add"),
        (chain_ops, "chain_ops"),
        (many_ops, "many_ops"),
        (relu_sigmoid, "relu_sigmoid"),
    ],
)
def test_aten_ops_match(fn, name):
    """Both pipelines produce the same sequence of ATen operations."""
    x, y = torch.randn(4, 4), torch.randn(4, 4)

    mini_gm = get_aten_ops_mini(fn, x, y)
    dynamo_gm = get_aten_ops_dynamo(fn, x, y)

    mini_ops = extract_op_list(mini_gm)
    dynamo_ops = extract_op_list(dynamo_gm)

    assert mini_ops == dynamo_ops, (
        f"ATen ops differ for {name}:\n"
        f"  minidynamo: {mini_ops}\n"
        f"  torch_dynamo: {dynamo_ops}"
    )


# ── Inductor source code parity tests ────────────────────────────────


@pytest.mark.parametrize(
    "fn,name",
    [
        (simple_add, "simple_add"),
        (chain_ops, "chain_ops"),
        (many_ops, "many_ops"),
        (relu_sigmoid, "relu_sigmoid"),
    ],
)
def test_inductor_kernels_match(fn, name):
    """Both pipelines produce the same Inductor C++ kernels."""
    x, y = torch.randn(4, 4), torch.randn(4, 4)

    mini_gm = get_aten_ops_mini(fn, x, y)
    dynamo_gm = get_aten_ops_dynamo(fn, x, y)

    mini_source = compile_and_get_source(mini_gm, (x, y))
    dynamo_source = compile_and_get_source(dynamo_gm, (x, y))

    mini_kernels = extract_kernel_bodies(mini_source)
    dynamo_kernels = extract_kernel_bodies(dynamo_source)

    assert mini_kernels and dynamo_kernels, (
        "No C++ kernels found; parity was not checked"
    )
    assert len(mini_kernels) == len(dynamo_kernels), (
        f"Different number of kernels for {name}: "
        f"mini={len(mini_kernels)}, dynamo={len(dynamo_kernels)}"
    )

    for i, (mk, dk) in enumerate(zip(mini_kernels, dynamo_kernels)):
        assert normalize_kernel(mk) == normalize_kernel(dk), (
            f"Kernel {i} differs for {name}:\n"
            f"  minidynamo:\n{mk}\n"
            f"  torch_dynamo:\n{dk}"
        )


# ── Correctness parity tests ─────────────────────────────────────────


@pytest.mark.parametrize(
    "fn,name",
    [
        (simple_add, "simple_add"),
        (chain_ops, "chain_ops"),
        (many_ops, "many_ops"),
        (relu_sigmoid, "relu_sigmoid"),
    ],
)
def test_inductor_results_match(fn, name):
    """Both Inductor-compiled versions produce the same numerical result."""
    x, y = torch.randn(8, 8), torch.randn(8, 8)

    mini_gm = get_aten_ops_mini(fn, x, y)
    dynamo_gm = get_aten_ops_dynamo(fn, x, y)

    # Compile runnable versions
    mini_compiled = compile_fx_inner(mini_gm, [x, y])
    dynamo_compiled = compile_fx_inner(dynamo_gm, [x, y])

    mini_result = mini_compiled([x, y])[0]
    dynamo_result = dynamo_compiled([x, y])[0]
    eager_result = fn(x, y)

    assert torch.allclose(mini_result, eager_result, atol=1e-5), (
        f"minidynamo+Inductor result doesn't match eager for {name}"
    )
    assert torch.allclose(dynamo_result, eager_result, atol=1e-5), (
        f"torch.compile+Inductor result doesn't match eager for {name}"
    )
    assert torch.allclose(mini_result, dynamo_result, atol=1e-6), (
        f"mini and dynamo Inductor results differ for {name}"
    )


def graph_signature(gm):
    """Compare operation wiring and constants as well as target names."""
    indices = {node: index for index, node in enumerate(gm.graph.nodes)}

    def encode(value):
        if isinstance(value, fx.Node):
            return ("node", indices[value])
        if isinstance(value, (tuple, list)):
            return tuple(encode(v) for v in value)
        if isinstance(value, dict):
            return tuple((k, encode(v)) for k, v in sorted(value.items()))
        return (type(value).__name__, repr(value))

    return [
        (
            node.op,
            str(node.target) if node.op != "placeholder" else "input",
            encode(node.args),
            encode(node.kwargs),
        )
        for node in gm.graph.nodes
    ]


@pytest.mark.parametrize("fn", [simple_add, chain_ops, many_ops, relu_sigmoid])
def test_complete_aten_graphs_match(fn):
    torch.manual_seed(0)
    x, y = torch.randn(4, 4), torch.randn(4, 4)
    assert graph_signature(get_aten_ops_mini(fn, x, y)) == graph_signature(
        get_aten_ops_dynamo(fn, x, y)
    )


def extract_triton_kernels(source):
    """Read the source strings passed to async_compile.triton, without executing them."""
    return [
        node.args[1].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "async_compile"
        and node.func.attr == "triton"
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "fn,size",
    [
        (simple_add, 32),
        (chain_ops, 32),
        (many_ops, 32),
        (relu_sigmoid, 32),
        (many_ops, 2048),
    ],
)
def test_cuda_graphs_kernels_and_results_match(fn, size):
    """Compare actual Triton code and graph wiring through the shared inference path."""
    import torch._inductor.config as config

    torch.manual_seed(0)
    with torch.inference_mode(), config.patch({"triton.cudagraphs": False}):
        inputs = tuple(torch.randn(size, size, device="cuda") for _ in range(2))
        mini_gm = get_aten_ops_mini(fn, *inputs)
        dynamo_gm = get_aten_ops_dynamo(fn, *inputs)
        assert graph_signature(mini_gm) == graph_signature(dynamo_gm)

        mini_compiled = compile_fx_inner(mini_gm, list(inputs))
        dynamo_compiled = compile_fx_inner(dynamo_gm, list(inputs))
        mini_kernels = extract_triton_kernels(mini_compiled.source_code)
        dynamo_kernels = extract_triton_kernels(dynamo_compiled.source_code)
        assert mini_kernels and dynamo_kernels, "No Triton kernels found"
        assert mini_kernels == dynamo_kernels

        for scale in (1, 0.75):
            fresh = tuple(x * scale for x in inputs)
            expected = fn(*fresh)
            for compiled in (mini_compiled, dynamo_compiled):
                torch.testing.assert_close(
                    compiled(list(fresh))[0], expected, rtol=1e-4, atol=1e-3
                )


def test_inductor_bridge_checks_inputs_and_globals():
    namespace = {"scale": 2}
    exec("def fn(x): return x * scale", namespace)
    fn = namespace["fn"]
    x = torch.randn(3, 3)
    compiled = minidynamo_to_inductor(fn, x)
    torch.testing.assert_close(compiled(x + 1), fn(x + 1))
    for changed in (torch.randn(4, 4), x.t(), x.double(), x.clone().requires_grad_()):
        with pytest.raises(ValueError, match="assumptions changed"):
            compiled(changed)
    with torch.no_grad(), pytest.raises(ValueError, match="assumptions changed"):
        compiled(x)
    namespace["scale"] = 3
    with pytest.raises(ValueError, match="assumptions changed"):
        compiled(x)


def test_inductor_bridge_rejects_training():
    with pytest.raises(NotImplementedError, match="inference-only"):
        minidynamo_to_inductor(
            simple_add, torch.randn(3, requires_grad=True), torch.randn(3)
        )


def test_fx_conversion_preserves_nested_arguments_and_kwargs():
    from minidynamo.graph import Graph

    graph = Graph()
    x, y = graph.placeholder("x"), graph.placeholder("y")
    out = graph.call_function(torch.cat, ((x, y),), {"dim": 1})
    graph.output(graph.call_method("sum", (out,), {"dim": 0}))
    gm = to_fx_graph_module(graph)
    a, b = torch.ones(2, 1), torch.full((2, 1), 3.0)
    torch.testing.assert_close(gm(a, b)[0], torch.cat((a, b), dim=1).sum(dim=0))
