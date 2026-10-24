"""Tests for the symbolic bytecode interpreter."""

import torch
import pytest
from minidynamo.symbolic_interpreter import SymbolicInterpreter
from minidynamo.compiler import compile_graph


# ── Helper to trace a function and return its graph ────────────────────


def trace(fn, *example_args):
    """Trace fn with example_args, return the captured Graph."""
    interp = SymbolicInterpreter(fn, example_args)
    return interp.run()


def opaque_helper(x):
    return torch.relu(x)


# ── Basic arithmetic ───────────────────────────────────────────────────


def test_add():
    def fn(x, y):
        return x + y

    g = trace(fn, torch.randn(3), torch.randn(3))
    ops = [(n.op, n.target) for n in g.nodes]
    assert ops == [
        ("placeholder", "x"),
        ("placeholder", "y"),
        ("call_function", torch.add),
        ("output", "output"),
    ]


def test_subtract():
    def fn(x, y):
        return x - y

    g = trace(fn, torch.randn(3), torch.randn(3))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert len(fn_nodes) == 1
    assert fn_nodes[0].target is torch.sub


def test_multiply():
    def fn(x, y):
        return x * y

    g = trace(fn, torch.randn(3), torch.randn(3))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert fn_nodes[0].target is torch.mul


def test_divide():
    def fn(x, y):
        return x / y

    g = trace(fn, torch.randn(3), torch.randn(3))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert fn_nodes[0].target is torch.div


def test_power():
    def fn(x):
        return x**2

    g = trace(fn, torch.randn(3))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert fn_nodes[0].target is torch.pow
    # The constant 2 should be in the args
    assert fn_nodes[0].args[1] == 2


# ── Tensor + constant operations ──────────────────────────────────────


def test_tensor_times_constant():
    def fn(x):
        return x * 2

    g = trace(fn, torch.randn(4))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert len(fn_nodes) == 1
    assert fn_nodes[0].target is torch.mul
    # args should be (x_node, 2) — the constant is inlined
    assert fn_nodes[0].args[0].name == "x"
    assert fn_nodes[0].args[1] == 2


def test_constant_plus_tensor():
    def fn(x):
        return 1 + x

    g = trace(fn, torch.randn(4))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert fn_nodes[0].target is torch.add
    assert fn_nodes[0].args[0] == 1  # constant first
    assert fn_nodes[0].args[1].name == "x"  # tensor second


# ── Chained operations ────────────────────────────────────────────────


def test_chain_add_mul_sum():
    """z = x + y; w = z * 2; return w.sum()"""

    def fn(x, y):
        z = x + y
        w = z * 2
        return w.sum()

    g = trace(fn, torch.randn(3, 4), torch.randn(3, 4))

    # Check structure: placeholder, placeholder, add, mul, sum, output
    op_types = [n.op for n in g.nodes]
    assert op_types == [
        "placeholder",
        "placeholder",
        "call_function",
        "call_function",
        "call_method",
        "output",
    ]

    # Check data flow: add feeds into mul, mul feeds into sum
    add_node = g.nodes[2]
    mul_node = g.nodes[3]
    sum_node = g.nodes[4]
    assert mul_node.args[0] is add_node  # mul takes add's output
    assert sum_node.args[0] is mul_node  # sum takes mul's output


def test_multi_step_computation():
    """a = x * x; b = a + y; c = b - x; return c"""

    def fn(x, y):
        a = x * x
        b = a + y
        c = b - x
        return c

    g = trace(fn, torch.randn(5), torch.randn(5))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert len(fn_nodes) == 3
    assert [n.target for n in fn_nodes] == [torch.mul, torch.add, torch.sub]


# ── Explicit torch function calls ─────────────────────────────────────


def test_torch_add():
    def fn(x, y):
        return torch.add(x, y)

    g = trace(fn, torch.randn(3), torch.randn(3))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert fn_nodes[0].target is torch.add


def test_torch_relu():
    def fn(x):
        return torch.relu(x)

    g = trace(fn, torch.randn(3))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert fn_nodes[0].target is torch.relu


def test_torch_sigmoid():
    def fn(x):
        return torch.sigmoid(x)

    g = trace(fn, torch.randn(3))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert fn_nodes[0].target is torch.sigmoid


def test_additional_supported_unary_torch_functions():
    def fn(x):
        return torch.abs(x) + torch.tanh(x) + torch.exp(x) + torch.sin(x) + torch.cos(x)

    g = trace(fn, torch.randn(3))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    targets = [n.target for n in fn_nodes]
    for torch_fn in [torch.abs, torch.tanh, torch.exp, torch.sin, torch.cos]:
        assert torch_fn in targets
    assert targets.count(torch.add) == 4


def test_explicit_supported_binary_torch_functions():
    def fn(x, y):
        return torch.sub(torch.mul(torch.add(x, y), y), torch.div(torch.pow(x, 2), y))

    x = torch.rand(3) + 1
    y = torch.rand(3) + 1
    g = trace(fn, x, y)
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert [n.target for n in fn_nodes] == [
        torch.add,
        torch.mul,
        torch.pow,
        torch.div,
        torch.sub,
    ]


def test_torch_matmul():
    def fn(x, y):
        return torch.matmul(x, y)

    g = trace(fn, torch.randn(3, 4), torch.randn(4, 5))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert fn_nodes[0].target is torch.matmul


def test_torch_chained():
    """z = torch.add(x, y); return torch.relu(z)"""

    def fn(x, y):
        z = torch.add(x, y)
        return torch.relu(z)

    g = trace(fn, torch.randn(3), torch.randn(3))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert len(fn_nodes) == 2
    assert fn_nodes[0].target is torch.add
    assert fn_nodes[1].target is torch.relu
    # relu's input should be add's output
    assert fn_nodes[1].args[0] is fn_nodes[0]


# ── Method calls ──────────────────────────────────────────────────────


def test_method_sum():
    def fn(x):
        return x.sum()

    g = trace(fn, torch.randn(3, 4))
    method_nodes = [n for n in g.nodes if n.op == "call_method"]
    assert len(method_nodes) == 1
    assert method_nodes[0].target == "sum"


def test_method_mean():
    def fn(x):
        return x.mean()

    g = trace(fn, torch.randn(3, 4))
    method_nodes = [n for n in g.nodes if n.op == "call_method"]
    assert method_nodes[0].target == "mean"


def test_method_after_operation():
    """return (x + y).sum()"""

    def fn(x, y):
        return (x + y).sum()

    g = trace(fn, torch.randn(3, 4), torch.randn(3, 4))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    method_nodes = [n for n in g.nodes if n.op == "call_method"]
    assert fn_nodes[0].target is torch.add
    assert method_nodes[0].target == "sum"
    # sum's self should be add's output
    assert method_nodes[0].args[0] is fn_nodes[0]


def test_global_python_callable_is_rejected():
    def fn(x):
        return opaque_helper(x)

    with pytest.raises(NotImplementedError, match="Unsupported global or callable"):
        trace(fn, torch.randn(3))


# ── Unary operations ──────────────────────────────────────────────────


def test_unary_negative():
    def fn(x):
        return -x

    g = trace(fn, torch.randn(3))
    fn_nodes = [n for n in g.nodes if n.op == "call_function"]
    assert fn_nodes[0].target is torch.neg


# ── Graph correctness (verify the graph computes the right thing) ─────


def test_graph_gives_correct_result():
    """Verify the captured graph produces the same output as eager execution."""

    def fn(x, y):
        z = x + y
        w = z * 2
        return w.sum()

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])

    # Eager result
    expected = fn(x, y)

    # Trace and compile the captured graph, then run the generated callable.
    graph = SymbolicInterpreter(fn, (x, y)).run()
    compiled_fn, _source = compile_graph(graph)

    assert torch.allclose(compiled_fn(x, y), expected)


# ── Error handling ────────────────────────────────────────────────────


def test_unsupported_bytecode_raises():
    """Control flow (if/for) should raise NotImplementedError."""

    def fn(x):
        if x.sum() > 0:  # This generates jump bytecodes
            return x
        return -x

    with pytest.raises(NotImplementedError, match="Unsupported bytecode"):
        trace(fn, torch.randn(3))


def test_keyword_calls_are_unsupported():
    def fn(x, y):
        return torch.add(x, y, alpha=0.5)

    with pytest.raises(NotImplementedError, match="Unsupported bytecode"):
        trace(fn, torch.randn(3), torch.randn(3))


def test_non_tensor_return_raises():
    """Returning a non-tensor should raise an error."""

    def fn(x):
        return 42

    with pytest.raises(RuntimeError, match="return a tensor"):
        trace(fn, torch.randn(3))


# ── Pretty printing ───────────────────────────────────────────────────


def test_graph_str_output():
    def fn(x, y):
        return x + y

    g = trace(fn, torch.randn(3), torch.randn(3))
    text = str(g)
    assert "x = placeholder" in text
    assert "y = placeholder" in text
    assert "add_0" in text
    assert "return" in text
