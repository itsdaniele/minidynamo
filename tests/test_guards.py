"""Tests for the guard system."""

import torch
from minidynamo.guards import Guard, GuardSet


# ── Individual guards ──────────────────────────────────────────────────


def test_guard_passes():
    g = Guard(lambda *args: args[0].shape == (3, 4), "args[0].shape == (3, 4)")
    assert g.check(torch.randn(3, 4)) is True


def test_guard_fails():
    g = Guard(lambda *args: args[0].shape == (3, 4), "args[0].shape == (3, 4)")
    assert g.check(torch.randn(5, 6)) is False


def test_guard_repr():
    g = Guard(lambda *args: True, "args[0].shape == (3, 4)")
    assert "args[0].shape == (3, 4)" in repr(g)


# ── GuardSet ───────────────────────────────────────────────────────────


def test_guard_set_all_pass():
    gs = GuardSet()
    gs.add(Guard(lambda *args: True, "always true"))
    gs.add(Guard(lambda *args: True, "also true"))
    assert gs.check_all(torch.randn(3)) is True


def test_guard_set_one_fails():
    gs = GuardSet()
    gs.add(Guard(lambda *args: True, "pass"))
    gs.add(Guard(lambda *args: False, "fail"))
    assert gs.check_all(torch.randn(3)) is False


def test_guard_set_empty_passes():
    """An empty guard set should pass (no conditions to violate)."""
    gs = GuardSet()
    assert gs.check_all() is True


def test_failing_guards():
    gs = GuardSet()
    gs.add(Guard(lambda *args: True, "pass"))
    gs.add(Guard(lambda *args: False, "this fails"))
    failing = gs.failing_guards(torch.randn(3))
    assert len(failing) == 1
    assert failing[0].description == "this fails"


def test_guard_set_repr():
    gs = GuardSet()
    gs.add(Guard(lambda *args: True, "args[0].shape == (3,)"))

    text = repr(gs)
    assert "GuardSet([" in text
    assert "args[0].shape == (3,)" in text


# ── GuardSet.from_example_inputs ───────────────────────────────────────


def test_from_example_inputs_single_tensor():
    x = torch.randn(3, 4)
    gs = GuardSet.from_example_inputs((x,))
    # A new tensor with the same metadata can reuse this entry.
    assert gs.check_all(torch.zeros_like(x))


def test_from_example_inputs_two_tensors():
    x = torch.randn(3, 4)
    y = torch.randn(5)
    gs = GuardSet.from_example_inputs((x, y))
    assert gs.check_all(torch.zeros_like(x), torch.zeros_like(y))


def test_guards_pass_same_metadata():
    """Guards pass when called with tensors of the same shape/dtype/device."""
    x = torch.randn(3, 4)
    y = torch.randn(5, 6)
    gs = GuardSet.from_example_inputs((x, y))

    # Same shapes → pass
    assert gs.check_all(torch.randn(3, 4), torch.randn(5, 6)) is True


def test_guards_fail_different_shape():
    """Guards fail when a tensor has a different shape."""
    x = torch.randn(3, 4)
    gs = GuardSet.from_example_inputs((x,))

    assert gs.check_all(torch.randn(3, 4)) is True
    assert gs.check_all(torch.randn(5, 6)) is False
    assert gs.check_all(torch.randn(3)) is False


def test_guards_fail_different_dtype():
    """Guards fail when a tensor has a different dtype."""
    x = torch.randn(3, 4)  # float32
    gs = GuardSet.from_example_inputs((x,))

    assert gs.check_all(torch.randn(3, 4)) is True  # float32 → pass
    assert (
        gs.check_all(torch.zeros(3, 4, dtype=torch.float64)) is False
    )  # float64 → fail


def test_different_argument_count_is_a_clean_miss():
    """A call with a different arity must miss the cache, not crash a lambda."""
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    gs = GuardSet.from_example_inputs((x, y))

    assert gs.check_all(x, y) is True
    assert gs.check_all(x) is False  # fewer args: clean miss, no IndexError
    assert gs.check_all(x, y, x) is False  # more args: also a clean miss
    assert gs.failing_guards(x) == gs.guards


def test_failing_guards_report_correct_argument_index():
    x = torch.randn(3, 4)
    y = torch.randn(5, 6)
    gs = GuardSet.from_example_inputs((x, y))

    failing = gs.failing_guards(torch.randn(3, 4), torch.randn(7, 8))

    assert any(g.description == "args[1].shape == (5, 6)" for g in failing)
    assert not any(g.description == "args[0].shape == (3, 4)" for g in failing)


def test_guards_descriptions_are_readable():
    """Guard descriptions should be human-readable."""
    x = torch.randn(3, 4)
    gs = GuardSet.from_example_inputs((x,))

    descriptions = [g.description for g in gs.guards]
    assert any("shape" in d for d in descriptions)
    assert any("dtype" in d for d in descriptions)
    assert any("device" in d for d in descriptions)


def test_guards_skip_non_tensor_args():
    """Non-tensor arguments should not generate guards."""
    gs = GuardSet.from_example_inputs((42, "hello", torch.randn(3)))
    # This helper only guards tensors; the public wrapper rejects other inputs.
    assert gs.check_all(42, "hello", torch.zeros(3))
