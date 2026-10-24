"""Tests for the compiler backend."""

import torch
from minidynamo.symbolic_interpreter import SymbolicInterpreter
from minidynamo.compiler import compile_graph


# ── Helper ─────────────────────────────────────────────────────────────


def trace_and_compile(fn, *example_args):
    """Trace a function and compile the graph. Returns (compiled_fn, source)."""
    graph = SymbolicInterpreter(fn, example_args).run()
    return compile_graph(graph)


# ── Correctness tests ─────────────────────────────────────────────────


def test_add():
    def fn(x, y):
        return x + y

    x, y = torch.randn(3), torch.randn(3)
    compiled_fn, _ = trace_and_compile(fn, x, y)
    assert torch.allclose(fn(x, y), compiled_fn(x, y))


def test_chain_add_mul_sum():
    def fn(x, y):
        z = x + y
        w = z * 2
        return w.sum()

    x, y = torch.randn(3, 4), torch.randn(3, 4)
    compiled_fn, _ = trace_and_compile(fn, x, y)
    assert torch.allclose(fn(x, y), compiled_fn(x, y))


def test_torch_relu():
    def fn(x):
        return torch.relu(x)

    x = torch.randn(5)
    compiled_fn, _ = trace_and_compile(fn, x)
    assert torch.allclose(fn(x), compiled_fn(x))


def test_additional_supported_torch_functions():
    def fn(x):
        return torch.sqrt(torch.abs(x) + 1) + torch.sin(x) + torch.cos(x)

    x = torch.randn(5)
    compiled_fn, _ = trace_and_compile(fn, x)
    assert torch.allclose(fn(x), compiled_fn(x))


def test_torch_add_relu():
    def fn(x, y):
        z = torch.add(x, y)
        return torch.relu(z)

    x, y = torch.randn(4), torch.randn(4)
    compiled_fn, _ = trace_and_compile(fn, x, y)
    assert torch.allclose(fn(x, y), compiled_fn(x, y))


def test_method_sum():
    def fn(x):
        return x.sum()

    x = torch.randn(3, 4)
    compiled_fn, _ = trace_and_compile(fn, x)
    assert torch.allclose(fn(x), compiled_fn(x))


def test_method_mean():
    def fn(x):
        return x.mean()

    x = torch.randn(3, 4)
    compiled_fn, _ = trace_and_compile(fn, x)
    assert torch.allclose(fn(x), compiled_fn(x))


def test_unary_negative():
    def fn(x):
        return -x

    x = torch.randn(5)
    compiled_fn, _ = trace_and_compile(fn, x)
    assert torch.allclose(fn(x), compiled_fn(x))


def test_complex_expression():
    """(x * y + y) / x — tests multiple ops chained."""

    def fn(x, y):
        return (x * y + y) / x

    x = torch.randn(4) + 1  # avoid division by zero
    y = torch.randn(4)
    compiled_fn, _ = trace_and_compile(fn, x, y)
    assert torch.allclose(fn(x, y), compiled_fn(x, y))


def test_tensor_times_scalar():
    def fn(x):
        return x * 3

    x = torch.randn(5)
    compiled_fn, _ = trace_and_compile(fn, x)
    assert torch.allclose(fn(x), compiled_fn(x))


def test_matmul():
    def fn(x, y):
        return torch.matmul(x, y)

    x, y = torch.randn(3, 4), torch.randn(4, 5)
    compiled_fn, _ = trace_and_compile(fn, x, y)
    assert torch.allclose(fn(x, y), compiled_fn(x, y))


# ── Works with different inputs (same shape) ──────────────────────────


def test_compiled_fn_works_with_different_data():
    """The compiled fn should work with any data of the same shape."""

    def fn(x, y):
        return x + y

    x1, y1 = torch.randn(3), torch.randn(3)
    compiled_fn, _ = trace_and_compile(fn, x1, y1)

    # Call with completely different data (same shape)
    x2, y2 = torch.randn(3), torch.randn(3)
    assert torch.allclose(fn(x2, y2), compiled_fn(x2, y2))


# ── Source code generation ─────────────────────────────────────────────


def test_source_contains_function_def():
    def fn(x, y):
        return x + y

    _, source = trace_and_compile(fn, torch.randn(3), torch.randn(3))
    assert source.startswith("def compiled_fn(x, y):")


def test_source_has_return():
    def fn(x):
        return x.sum()

    _, source = trace_and_compile(fn, torch.randn(3))
    assert "return" in source


def test_source_uses_closure_vars():
    """Function calls should use closure variables like __fn_add_0."""

    def fn(x, y):
        return x + y

    _, source = trace_and_compile(fn, torch.randn(3), torch.randn(3))
    assert "__fn_add_0" in source


def test_source_method_call_dot_syntax():
    """Method calls should use dot syntax: x.sum()."""

    def fn(x):
        return x.sum()

    _, source = trace_and_compile(fn, torch.randn(3))
    assert ".sum()" in source
