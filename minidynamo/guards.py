"""
Guards — conditions under which compiled code can be reused.

When we trace a function with specific inputs, the resulting graph makes
assumptions about those inputs: their shapes, dtypes, and devices. Guards
encode these assumptions as simple boolean checks.

On each subsequent call, we run the guards against the new inputs:
  - All guards pass → use the cached compiled function (fast path)
  - Any guard fails → retrace and compile from scratch (slow path)

In real TorchDynamo, guards are much more extensive — they check type IDs,
object identity, dict versions, global variable values, tensor strides,
and more. We check tensor metadata and execution modes. The interpreter
adds guards for global bindings and torch attributes read during tracing.

Example:
  First call:  fn(tensor(shape=[3,4], dtype=float32, device=cpu))
  Guards:       args[0].shape == (3, 4)
                args[0].dtype == torch.float32
                args[0].device == cpu

  Second call: fn(tensor(shape=[3,4], dtype=float32, device=cpu))
  → All guards pass → reuse compiled code ✓

  Third call:  fn(tensor(shape=[5,6], dtype=float32, device=cpu))
  → Shape guard fails → retrace ✗
"""

import torch


def execution_context():
    """Ambient state that can change tracing and backend behavior."""
    return (
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.get_default_dtype(),
        # An unindexed device in x.to("cuda") resolves against this state.
        # CUDA's initial device is logical device 0. Treat lazy initialization
        # as unchanged state, without initializing CUDA just to guard CPU code.
        torch.cuda.current_device() if torch.cuda.is_initialized() else 0,
        *(
            (torch.is_autocast_enabled(device), torch.get_autocast_dtype(device))
            for device in ("cpu", "cuda")
        ),
    )


class Guard:
    """
    A single boolean check on the inputs.

    Fields:
        check_fn    — callable(*args) -> bool
        description — human-readable string for debugging (e.g. "args[0].shape == (3, 4)")
    """

    def __init__(self, check_fn, description):
        self.check_fn = check_fn
        self.description = description

    def check(self, *args):
        return self.check_fn(*args)

    def __repr__(self):
        return f"Guard({self.description})"


class GuardSet:
    """
    A collection of guards for one compiled function.

    Created from the example inputs used during tracing. Checks metadata
    and ambient execution modes; tracing adds guards for global bindings.
    """

    def __init__(self):
        self.guards = []
        # Number of arguments seen during tracing. Checked before the
        # per-tensor guards so that lambdas indexing args[i] never see a
        # call with a different number of arguments.
        self.n_args = None

    def add(self, guard):
        self.guards.append(guard)

    def check_all(self, *args):
        """Return True if ALL guards pass for the given arguments."""
        if self.n_args is not None and len(args) != self.n_args:
            return False
        return all(g.check(*args) for g in self.guards)

    def failing_guards(self, *args):
        """Return list of guards that fail — useful for debugging recompilations."""
        if self.n_args is not None and len(args) != self.n_args:
            # Different arity: this cache entry cannot apply at all.
            return list(self.guards)
        return [g for g in self.guards if not g.check(*args)]

    def __repr__(self):
        lines = ["GuardSet(["]
        for g in self.guards:
            lines.append(f"  {g.description}")
        lines.append("])")
        return "\n".join(lines)

    @classmethod
    def from_example_inputs(cls, example_args):
        """
        Create guards from the example inputs used during tracing.

        For each tensor argument, we create guards asserting its shape,
        dtype, and device match what we saw during tracing.
        """
        guard_set = cls()
        guard_set.n_args = len(example_args)

        for i, arg in enumerate(example_args):
            if isinstance(arg, torch.Tensor):
                # Shape guard
                expected_shape = tuple(arg.shape)
                guard_set.add(
                    Guard(
                        # Use default args to capture i and expected_shape by value
                        # (otherwise the lambda would capture the loop variable by reference)
                        lambda *args, idx=i, s=expected_shape: (
                            type(args[idx]) is torch.Tensor
                            and tuple(args[idx].shape) == s
                        ),
                        f"args[{i}].shape == {expected_shape}",
                    )
                )

                # Dtype guard
                expected_dtype = arg.dtype
                guard_set.add(
                    Guard(
                        lambda *args, idx=i, d=expected_dtype: (
                            type(args[idx]) is torch.Tensor and args[idx].dtype == d
                        ),
                        f"args[{i}].dtype == {expected_dtype}",
                    )
                )

                # Device guard
                expected_device = arg.device
                guard_set.add(
                    Guard(
                        lambda *args, idx=i, dev=expected_device: (
                            type(args[idx]) is torch.Tensor and args[idx].device == dev
                        ),
                        f"args[{i}].device == {expected_device}",
                    )
                )

                for attr in (
                    "stride",
                    "storage_offset",
                    "requires_grad",
                    "is_conj",
                    "is_neg",
                ):
                    expected = getattr(arg, attr)
                    expected = expected() if callable(expected) else expected

                    def check(*args, idx=i, name=attr, value=expected):
                        if (
                            type(args[idx]) is not torch.Tensor
                            or args[idx].layout != torch.strided
                        ):
                            return False
                        actual = getattr(args[idx], name)
                        return (actual() if callable(actual) else actual) == value

                    guard_set.add(Guard(check, f"args[{i}].{attr} == {expected}"))

        context = execution_context()
        guard_set.add(
            Guard(
                lambda *args: execution_context() == context,
                "grad, inference, autocast modes, default dtype and CUDA device are unchanged",
            )
        )

        return guard_set
