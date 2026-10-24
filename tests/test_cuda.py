"""CUDA parity across the toy backends, plus gradients and cache behavior."""

import subprocess
import sys

import pytest
import torch
import minidynamo
from minidynamo.inductor import minidynamo_to_inductor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def pointwise(x, y):
    return torch.tanh(torch.relu(x + y) * 0.5) - y


def reduction(x, y):
    return ((x + y) * 2).sum()


def matmul(x, y):
    return torch.relu(torch.matmul(x, y))


@pytest.mark.parametrize("fn", [pointwise, reduction, matmul])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["python", "jit", "inductor"])
def test_cuda_backends(fn, dtype, backend):
    torch.manual_seed(7)
    with torch.inference_mode():
        x, y = [torch.randn(32, 32, device="cuda", dtype=dtype) for _ in range(2)]
        compiled = (
            minidynamo_to_inductor(fn, x, y)
            if backend == "inductor"
            else minidynamo.compile(fn, backend=backend)
        )
        for scale in (1, 0.75):
            inputs = (x * scale, y * scale)
            # Fusion may eliminate reduced-precision intermediate rounding.
            tolerance = (
                3e-2
                if dtype == torch.bfloat16
                else 3e-3
                if dtype == torch.float16
                else 1e-4
            )
            torch.testing.assert_close(
                compiled(*inputs), fn(*inputs), rtol=tolerance, atol=tolerance
            )


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_cuda_gradients_and_recompilation(backend):
    compiled = minidynamo.compile(pointwise, backend=backend)
    for shape in ((3, 4), (5, 6), (3, 4)):
        x = torch.randn(*shape, device="cuda", requires_grad=True)
        y = torch.randn(*shape, device="cuda", requires_grad=True)
        result = compiled(x, y)
        expected = pointwise(x, y)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(
            torch.autograd.grad(result.sum(), (x, y)),
            torch.autograd.grad(expected.sum(), (x, y)),
        )
    assert len(compiled._cache) == 2


@pytest.mark.parametrize("backend", ["python", "jit", "inductor"])
def test_cuda_noncontiguous_inputs(backend):
    x, y = [torch.randn(4, 4, device="cuda").t() for _ in range(2)]
    compiled = (
        minidynamo_to_inductor(pointwise, x, y)
        if backend == "inductor"
        else minidynamo.compile(pointwise, backend=backend)
    )
    torch.testing.assert_close(compiled(x, y), pointwise(x, y))


def test_cuda_current_device_is_guarded(monkeypatch):
    def fn(x):
        return x.to("cuda") + 1

    # A transfer to the current device is a no-op during this trace. On a
    # later call, a changed current device would require a different graph.
    device = torch.cuda.current_device()
    x = torch.ones(3, device="cuda")
    compiled = minidynamo_to_inductor(fn, x)
    torch.testing.assert_close(compiled(x), fn(x))
    # Simulate a caller selecting another device, without requiring two GPUs.
    # The wrapper must reject the change before entering compiled device code.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: device + 1)
    with pytest.raises(ValueError, match="assumptions changed"):
        compiled(x)


def test_cpu_inductor_bridge_survives_lazy_cuda_initialization():
    # Inductor can initialize CUDA while compiling an entirely CPU graph.
    # Start a fresh interpreter: preceding CUDA tests would hide this case.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch\n"
            "from minidynamo.inductor import minidynamo_to_inductor\n"
            "assert not torch.cuda.is_initialized()\n"
            "x = torch.ones(3)\n"
            "fn = lambda x: (x + 1).sum()\n"
            "compiled = minidynamo_to_inductor(fn, x)\n"
            "torch.testing.assert_close(compiled(x), fn(x))\n",
        ],
        check=True,
        timeout=120,
    )
