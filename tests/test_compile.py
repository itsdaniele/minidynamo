"""Tests for the compile() decorator — the full pipeline."""

import torch
import pytest
import minidynamo


# ── Correctness ────────────────────────────────────────────────────────


def test_basic_add():
    @minidynamo.compile
    def fn(x, y):
        return x + y

    x, y = torch.randn(3), torch.randn(3)
    assert torch.allclose(fn(x, y), x + y)


def test_chain():
    @minidynamo.compile
    def fn(x, y):
        z = x + y
        w = z * 2
        return w.sum()

    x, y = torch.randn(3, 4), torch.randn(3, 4)
    expected = (x + y).mul(2).sum()
    assert torch.allclose(fn(x, y), expected)


def test_torch_functions():
    @minidynamo.compile
    def fn(x):
        return torch.relu(torch.sigmoid(x))

    x = torch.randn(5)
    assert torch.allclose(fn(x), torch.relu(torch.sigmoid(x)))


def test_method_call():
    @minidynamo.compile
    def fn(x):
        return x.sum()

    x = torch.randn(4, 5)
    assert torch.allclose(fn(x), x.sum())


def test_complex_computation():
    @minidynamo.compile
    def fn(x, y):
        a = x * y
        b = a + y
        c = torch.relu(b)
        return c.mean()

    x, y = torch.randn(3, 4), torch.randn(3, 4)
    expected = torch.relu(x * y + y).mean()
    assert torch.allclose(fn(x, y), expected)


# ── Caching behavior ──────────────────────────────────────────────────


def test_cache_hit_same_shape():
    """Same shape on second call → should reuse cache (1 entry)."""

    @minidynamo.compile
    def fn(x):
        return x * 2

    fn(torch.randn(3, 4))
    fn(torch.randn(3, 4))  # same shape → cache hit
    assert len(fn._cache) == 1


def test_cache_miss_different_shape():
    """Different shape → guard fails → retrace → 2 cache entries."""

    @minidynamo.compile
    def fn(x):
        return x * 2

    fn(torch.randn(3, 4))
    fn(torch.randn(5, 6))  # different shape → cache miss
    assert len(fn._cache) == 2


def test_cache_hit_after_miss():
    """After recompilation, original shape still works from cache."""

    @minidynamo.compile
    def fn(x):
        return x + 1

    x1 = torch.randn(3)
    x2 = torch.randn(5)

    r1 = fn(x1)  # compile for shape (3,)
    fn(x2)  # compile for shape (5,)
    r1_again = fn(x1)  # should hit first cache entry

    assert torch.allclose(r1, r1_again)
    assert len(fn._cache) == 2


def test_cache_miss_different_dtype():
    """Different dtype → guard fails → retrace."""

    @minidynamo.compile
    def fn(x):
        return x * 2

    fn(torch.randn(3, dtype=torch.float32))
    fn(torch.randn(3, dtype=torch.float64))
    assert len(fn._cache) == 2


def test_non_tensor_runtime_args_are_explicitly_unsupported():
    """Runtime constants are intentionally out of scope for the decorator."""

    @minidynamo.compile
    def fn(x, scale):
        return x * scale

    with pytest.raises(TypeError, match="positional tensor arguments"):
        fn(torch.randn(3), 2)


def test_keyword_arguments_are_explicitly_unsupported():
    @minidynamo.compile
    def fn(x):
        return x * 2

    with pytest.raises(TypeError):
        fn(x=torch.randn(3))


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Unsupported backend"):
        minidynamo.compile(backend="inductor")


# ── Multiple calls produce correct results ─────────────────────────────


def test_multiple_calls_correct():
    """Compiled function works correctly across many calls."""

    @minidynamo.compile
    def fn(x, y):
        return x + y

    for _ in range(10):
        x, y = torch.randn(4), torch.randn(4)
        assert torch.allclose(fn(x, y), x + y)

    # All same shape → should be 1 cache entry
    assert len(fn._cache) == 1


# ── Wrapper preserves function metadata ────────────────────────────────


def test_wrapper_preserves_name():
    @minidynamo.compile
    def my_function(x):
        return x * 2

    assert my_function.__name__ == "my_function"


def test_original_fn_accessible():
    @minidynamo.compile
    def fn(x):
        return x * 2

    assert fn._original is not None
    assert callable(fn._original)
