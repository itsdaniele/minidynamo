"""Behavioral regressions found while reviewing the publication draft."""

import pytest
import torch

import minidynamo
from minidynamo.compiler import compile_graph
from minidynamo.graph import Graph
from minidynamo.guards import GuardSet


def make_function(source, **bindings):
    namespace = {"torch": torch, **bindings}
    exec(source, namespace)
    return namespace["fn"], namespace


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_rebound_global_invalidates_cache(backend):
    fn, namespace = make_function("def fn(x): return x * scale", scale=2)
    compiled = minidynamo.compile(fn, backend=backend)
    x = torch.ones(3)
    torch.testing.assert_close(compiled(x), x * 2)
    namespace["scale"] = 3
    torch.testing.assert_close(compiled(x), x * 3)
    assert len(compiled._cache) == 2


def test_rebound_torch_alias_invalidates_cache():
    fn, namespace = make_function("def fn(x): return op(x)", op=torch.sin)
    compiled = minidynamo.compile(fn)
    x = torch.ones(3)
    torch.testing.assert_close(compiled(x), x.sin())
    namespace["op"] = torch.cos
    torch.testing.assert_close(compiled(x), x.cos())


def test_global_none_is_not_a_missing_name():
    fn, _ = make_function("def fn(x): return x.sum(axis)", axis=None)
    x = torch.ones(2, 3)
    torch.testing.assert_close(minidynamo.compile(fn)(x), fn(x))


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_function_builtin_namespace_is_used_and_guarded(backend):
    builtins = {"float": int}
    fn, namespace = make_function(
        "def fn(x): return x * float(2.5)", __builtins__=builtins
    )
    compiled = minidynamo.compile(fn, backend=backend)
    x = torch.ones(3)
    torch.testing.assert_close(compiled(x), x * 2)
    # Replacing the globals entry does not rebind an existing function's builtins.
    namespace["__builtins__"] = {"float": float}
    torch.testing.assert_close(compiled(x), fn(x))
    assert len(compiled._cache) == 1
    builtins["float"] = float
    torch.testing.assert_close(compiled(x), x * 2.5)
    assert len(compiled._cache) == 2
    # A new global binding takes precedence over the function's builtins.
    namespace["float"] = int
    torch.testing.assert_close(compiled(x), x * 2)


@pytest.mark.parametrize("backend", ["python", "jit"])
@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("method", ["sum", "numel"])
def test_overridden_tensor_methods_rejected_without_execution(
    backend, warm_cache, method
):
    fn, _ = make_function(f"def fn(x): return x * x.{method}()")
    compiled = minidynamo.compile(fn, backend=backend)
    x = torch.ones(3)
    if warm_cache:
        compiled(x)
    calls = []

    def override():
        calls.append(1)
        return 7

    setattr(x, method, override)
    with pytest.raises(TypeError, match="without custom attributes"):
        compiled(x)
    assert not calls


@pytest.mark.parametrize(
    "statement",
    [
        "x.add_(1)",
        "x.zero_()",
        "print(x)",
        "torch.manual_seed(4)",
        "torch.randn(3)",
        "torch.rand_like(x)",
        "torch.add(1, 2)",
    ],
)
def test_side_effects_rejected_before_execution(statement, capsys):
    fn, _ = make_function(f"def fn(x):\n    {statement}\n    return x")
    x = torch.ones(3)
    state = torch.random.get_rng_state().clone()
    with pytest.raises(NotImplementedError):
        minidynamo.compile(fn)(x)
    torch.testing.assert_close(x, torch.ones(3))
    assert torch.equal(torch.random.get_rng_state(), state)
    assert capsys.readouterr().out == ""


def test_opaque_helper_is_rejected_without_calling_it():
    calls = []

    def helper(x):
        calls.append(x)
        return x + 1

    fn, _ = make_function("def fn(x): return helper(x)", helper=helper)
    with pytest.raises(NotImplementedError):
        minidynamo.compile(fn)(torch.ones(3))
    assert not calls


@pytest.mark.parametrize(
    "expression", ["float(x.sum())", "int(x.sum())", "bool(x.sum())", "x.sum().item()"]
)
def test_tensor_data_cannot_be_frozen_as_python_constant(expression):
    fn, _ = make_function(f"def fn(x): return x * {expression}")
    with pytest.raises(NotImplementedError):
        minidynamo.compile(fn)(torch.ones(3))


@pytest.mark.parametrize("parameter", ["add_0", "__fn_add_0", "compiled_fn"])
def test_codegen_names_do_not_shadow_inputs(parameter):
    fn, _ = make_function(f"def fn(x, {parameter}): return (x + 1) * {parameter}")
    args = (torch.ones(3), torch.full((3,), 4.0))
    torch.testing.assert_close(minidynamo.compile(fn)(*args), fn(*args))


@pytest.mark.parametrize(
    "expression",
    [
        "x.to(torch.float64)",
        "x.reshape(x.shape)",
        "x * x.dim()",
        "x * x.numel()",
        "x * len(x.shape)",
        "2 - x",
        "2 / x",
        "2 ** x",
    ],
)
@pytest.mark.parametrize("backend", ["python", "jit"])
def test_metadata_constants_and_reflected_arithmetic(expression, backend):
    fn, _ = make_function(f"def fn(x): return {expression}")
    compiled = minidynamo.compile(fn, backend=backend)
    for shape in ((2, 3), (3, 4)):
        x = torch.rand(*shape) + 0.5
        torch.testing.assert_close(compiled(x), fn(x))


@pytest.mark.parametrize("method", [False, True])
def test_graph_kwargs_are_executed(method):
    graph = Graph()
    x, y = graph.placeholder("x"), graph.placeholder("y")
    result = (
        graph.call_method("add", (x, y), {"alpha": 2})
        if method
        else graph.call_function(torch.add, (x, y), {"alpha": 2})
    )
    graph.output(result)
    compiled, _ = compile_graph(graph)
    torch.testing.assert_close(
        compiled(torch.ones(3), torch.ones(3)), torch.full((3,), 3.0)
    )


def test_nested_node_arguments_in_codegen():
    graph = Graph()
    x, y = graph.placeholder("x"), graph.placeholder("y")
    graph.output(graph.call_function(torch.cat, ((x, y),), {"dim": 0}))
    compiled, _ = compile_graph(graph)
    torch.testing.assert_close(
        compiled(torch.ones(2), torch.zeros(2)), torch.tensor([1.0, 1.0, 0.0, 0.0])
    )


def test_nonfinite_constant_codegen():
    graph = Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_function(torch.mul, (x, float("inf"))))
    compiled, _ = compile_graph(graph)
    assert torch.isposinf(compiled(torch.ones(1))).all()


@pytest.mark.parametrize("warm_cache", [False, True])
def test_wrong_arity_is_a_type_error(warm_cache):
    def fn(x, y):
        return x + y

    compiled = minidynamo.compile(fn)
    x = torch.ones(3)
    if warm_cache:
        compiled(x, x)
    for args in ((x,), (x, x, x)):
        with pytest.raises(TypeError, match="expects 2 positional arguments"):
            compiled(*args)


def test_guards_reject_layout_gradient_and_context_changes():
    x = torch.ones(2, 2)
    guards = GuardSet.from_example_inputs((x,))
    assert not guards.check_all(x.t())
    assert not guards.check_all(x.clone().requires_grad_())
    assert not guards.check_all(None)
    with torch.no_grad():
        assert not guards.check_all(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert not guards.check_all(x)


def test_python_backend_preserves_autograd():
    def fn(x):
        return (x * x).sum()

    compiled = minidynamo.compile(fn)
    for _ in range(2):
        x = torch.randn(3, requires_grad=True)
        compiled(x).backward()
        torch.testing.assert_close(x.grad, 2 * x)


def test_default_dtype_specialization_is_guarded():
    def fn(x):
        y = x + 0.5
        return x.to(y.dtype)

    compiled = minidynamo.compile(fn)
    x = torch.ones(3, dtype=torch.int64)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        torch.testing.assert_close(compiled(x), fn(x))
        torch.set_default_dtype(torch.float64)
        torch.testing.assert_close(compiled(x), fn(x))
        assert len(compiled._cache) == 2
    finally:
        torch.set_default_dtype(original_dtype)


def test_conjugate_and_negative_views_do_not_reuse_input_guards():
    x = torch.ones(3, dtype=torch.complex64)
    guards = GuardSet.from_example_inputs((x,))
    assert not guards.check_all(x.conj())
    assert not guards.check_all(torch._neg_view(x))
