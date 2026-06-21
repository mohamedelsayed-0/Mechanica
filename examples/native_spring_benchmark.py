"""Compare pure Torch and optional C++ spring kernels."""

import os
from pathlib import Path
from time import perf_counter

import torch

os.environ.setdefault(
    "MECHANICA_NATIVE_BUILD_DIR",
    str(Path(__file__).resolve().parents[1] / ".pytest_cache" / "torch_extensions"),
)

from mechanica import NativeExtensionUnavailable, hooke_spring_force, native_spring_status


def time_kernel(fn, *, warmup: int = 5, runs: int = 25) -> float:
    for _ in range(warmup):
        fn()
    start = perf_counter()
    for _ in range(runs):
        fn()
    return (perf_counter() - start) / runs


bodies = 512
springs = 2048
positions = torch.randn(bodies, 3)
velocities = torch.randn(bodies, 3)
edges = torch.randint(0, bodies, (springs, 2))
rest_lengths = torch.full((springs,), 0.75)
stiffness = torch.full((springs,), 15.0)


def torch_kernel() -> torch.Tensor:
    return hooke_spring_force(
        positions,
        edges,
        rest_lengths,
        stiffness,
        velocities=velocities,
        damping=0.05,
        use_native=False,
    )


def native_kernel() -> torch.Tensor:
    return hooke_spring_force(
        positions,
        edges,
        rest_lengths,
        stiffness,
        velocities=velocities,
        damping=0.05,
        use_native=True,
    )


torch_time = time_kernel(torch_kernel)
print(f"torch spring kernel: {torch_time * 1e3:.3f} ms")

available, reason = native_spring_status()
if not available:
    print("native spring kernel unavailable:")
    print(reason)
else:
    try:
        native_time = time_kernel(native_kernel)
    except NativeExtensionUnavailable as exc:
        print("native spring kernel unavailable:")
        print(exc)
    else:
        print(f"native spring kernel: {native_time * 1e3:.3f} ms")
        print(f"speedup: {torch_time / native_time:.2f}x")
        assert torch.allclose(torch_kernel(), native_kernel(), atol=1e-5)
