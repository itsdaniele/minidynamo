"""Tests for the JIT trace compiler backend."""

import torch
import pytest
import minidynamo
from minidynamo.symbolic_interpreter import SymbolicInterpreter
from minidynamo.compiler import compile_graph_jit


HAS_MPS = torch.backends.mps.is_available()
HAS_CUDA = torch.cuda.is_available()


# ── Helper ─────────────────────────────────────────────────────────────


def trace_and_jit(fn, *example_args):
    """Trace a function and JIT-compile the graph."""
    graph = SymbolicInterpreter(fn, example_args).run()
    return compile_graph_jit(graph, example_args)


# ── Correctness: compile_graph_jit on CPU ──────────────────────────────


def test_jit_add():
    def fn(x, y):
        return x + y

    x, y = torch.randn(3), torch.randn(3)
    jit_fn, _ = trace_and_jit(fn, x, y)
    assert torch.allclose(fn(x, y), jit_fn(x, y))


def test_jit_chain():
    def fn(x, y):
        z = x + y
        w = z * 2
        return w.sum()

    x, y = torch.randn(3, 4), torch.randn(3, 4)
    jit_fn, _ = trace_and_jit(fn, x, y)
    assert torch.allclose(fn(x, y), jit_fn(x, y))


def test_jit_torch_relu():
    def fn(x):
        return torch.relu(x)

    x = torch.randn(5)
    jit_fn, _ = trace_and_jit(fn, x)
    assert torch.allclose(fn(x), jit_fn(x))


def test_jit_method_sum():
    def fn(x):
        return x.sum()

    x = torch.randn(4, 5)
    jit_fn, _ = trace_and_jit(fn, x)
    assert torch.allclose(fn(x), jit_fn(x))


def test_jit_complex():
    def fn(x, y):
        a = x * y
        b = a + y
        c = torch.relu(b)
        d = torch.sigmoid(c)
        return d.mean()

    x, y = torch.randn(3, 4), torch.randn(3, 4)
    jit_fn, _ = trace_and_jit(fn, x, y)
    assert torch.allclose(fn(x, y), jit_fn(x, y))


def test_jit_matmul():
    def fn(x, y):
        return torch.matmul(x, y)

    x, y = torch.randn(4, 3), torch.randn(3, 5)
    jit_fn, _ = trace_and_jit(fn, x, y)
    assert torch.allclose(fn(x, y), jit_fn(x, y))


def test_jit_works_with_different_data():
    """JIT function works with different data of same shape."""

    def fn(x, y):
        return x + y

    x, y = torch.randn(4), torch.randn(4)
    jit_fn, _ = trace_and_jit(fn, x, y)

    x2, y2 = torch.randn(4), torch.randn(4)
    assert torch.allclose(fn(x2, y2), jit_fn(x2, y2))


def test_jit_returns_source():
    """compile_graph_jit returns the intermediate Python source."""

    def fn(x, y):
        return x + y

    x, y = torch.randn(3), torch.randn(3)
    _, source = trace_and_jit(fn, x, y)
    assert "def compiled_fn" in source


# ── Decorator with backend="jit" ──────────────────────────────────────


def test_decorator_jit_backend():
    @minidynamo.compile(backend="jit")
    def fn(x, y):
        return x + y

    x, y = torch.randn(3), torch.randn(3)
    assert torch.allclose(fn(x, y), x + y)


def test_decorator_jit_caching():
    @minidynamo.compile(backend="jit")
    def fn(x):
        return x * 2

    fn(torch.randn(3))
    fn(torch.randn(3))  # same shape → cache hit
    assert len(fn._cache) == 1


def test_decorator_jit_recompilation():
    @minidynamo.compile(backend="jit")
    def fn(x):
        return x * 2

    fn(torch.randn(3))
    fn(torch.randn(5))  # different shape → retrace
    assert len(fn._cache) == 2


def test_decorator_jit_backend_attribute():
    @minidynamo.compile(backend="jit")
    def fn(x):
        return x * 2

    assert fn._backend == "jit"


def test_decorator_default_backend_attribute():
    @minidynamo.compile
    def fn(x):
        return x * 2

    assert fn._backend == "python"


# ── MPS tests (skipped if MPS not available) ──────────────────────────


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_jit_on_mps():
    def fn(x, y):
        return x + y

    x = torch.randn(3, 4, device="mps")
    y = torch.randn(3, 4, device="mps")
    jit_fn, _ = trace_and_jit(fn, x, y)
    result = jit_fn(x, y)
    expected = fn(x, y)
    assert result.device.type == "mps"
    assert torch.allclose(result.cpu(), expected.cpu())


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_jit_chain_on_mps():
    def fn(x, y):
        z = x + y
        w = z * 2
        return torch.relu(w).sum()

    x = torch.randn(32, 32, device="mps")
    y = torch.randn(32, 32, device="mps")
    jit_fn, _ = trace_and_jit(fn, x, y)
    result = jit_fn(x, y)
    expected = fn(x, y)
    assert torch.allclose(result.cpu(), expected.cpu())


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_decorator_jit_on_mps():
    @minidynamo.compile(backend="jit")
    def fn(x, y):
        return torch.relu(x + y).mean()

    x = torch.randn(16, 16, device="mps")
    y = torch.randn(16, 16, device="mps")
    result = fn(x, y)
    expected = torch.relu(x + y).mean()
    assert torch.allclose(result.cpu(), expected.cpu())


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_python_backend_on_mps():
    """The default python backend also works with MPS tensors."""

    @minidynamo.compile
    def fn(x, y):
        return x + y

    x = torch.randn(4, device="mps")
    y = torch.randn(4, device="mps")
    result = fn(x, y)
    expected = x + y
    assert result.device.type == "mps"
    assert torch.allclose(result.cpu(), expected.cpu())


# ── CUDA tests (skipped if CUDA not available) ─────────────────────────


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_jit_on_cuda():
    def fn(x, y):
        return x + y

    x = torch.randn(3, 4, device="cuda")
    y = torch.randn(3, 4, device="cuda")
    jit_fn, _ = trace_and_jit(fn, x, y)
    result = jit_fn(x, y)
    expected = fn(x, y)
    assert result.device.type == "cuda"
    assert torch.allclose(result.cpu(), expected.cpu())


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_jit_chain_on_cuda():
    def fn(x, y):
        z = x + y
        w = z * 2
        return torch.relu(w).sum()

    x = torch.randn(32, 32, device="cuda")
    y = torch.randn(32, 32, device="cuda")
    jit_fn, _ = trace_and_jit(fn, x, y)
    result = jit_fn(x, y)
    expected = fn(x, y)
    assert torch.allclose(result.cpu(), expected.cpu())


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_decorator_jit_on_cuda():
    @minidynamo.compile(backend="jit")
    def fn(x, y):
        return torch.relu(x + y).mean()

    x = torch.randn(16, 16, device="cuda")
    y = torch.randn(16, 16, device="cuda")
    result = fn(x, y)
    expected = torch.relu(x + y).mean()
    assert torch.allclose(result.cpu(), expected.cpu())


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_python_backend_on_cuda():
    """The default python backend also works with CUDA tensors."""

    @minidynamo.compile
    def fn(x, y):
        return x + y

    x = torch.randn(4, device="cuda")
    y = torch.randn(4, device="cuda")
    result = fn(x, y)
    expected = x + y
    assert result.device.type == "cuda"
    assert torch.allclose(result.cpu(), expected.cpu())
