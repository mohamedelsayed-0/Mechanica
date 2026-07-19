"""Optional C++/Torch extension helpers."""

from __future__ import annotations

from functools import lru_cache
import importlib
import os
from pathlib import Path
from typing import Any

import torch

Tensor = torch.Tensor

_NATIVE_ENV = "MECHANICA_USE_NATIVE"
_VERBOSE_ENV = "MECHANICA_NATIVE_VERBOSE"
_BUILD_DIR_ENV = "MECHANICA_NATIVE_BUILD_DIR"
_TRUTHY = {"1", "true", "yes", "on"}


class NativeExtensionUnavailable(RuntimeError):
    """Raised when an optional native extension cannot be loaded."""


def native_springs_requested() -> bool:
    """Return whether spring kernels should try the optional native backend."""
    return os.environ.get(_NATIVE_ENV, "").strip().lower() in _TRUTHY


def _as_tensor_like(value: float | Tensor, like: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


@lru_cache(maxsize=1)
def _load_spring_extension() -> Any:
    try:
        return importlib.import_module("mechanica._mechanica_native")
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - depends on local torch install
        raise NativeExtensionUnavailable("torch C++ extension loader is unavailable") from exc

    source = Path(__file__).with_name("native") / "spring.cpp"
    if not source.exists():
        raise NativeExtensionUnavailable(f"native source file is missing: {source}")

    extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"]
    verbose = os.environ.get(_VERBOSE_ENV, "").strip().lower() in _TRUTHY
    build_directory = os.environ.get(_BUILD_DIR_ENV)
    if build_directory:
        Path(build_directory).mkdir(parents=True, exist_ok=True)

    try:
        return load(
            name="mechanica_native_springs",
            sources=[str(source)],
            build_directory=build_directory,
            extra_cflags=extra_cflags,
            with_cuda=False,
            verbose=verbose,
        )
    except Exception as exc:  # pragma: no cover - depends on compiler availability
        msg = f"could not compile or load the mechanica spring C++ extension: {exc}"
        raise NativeExtensionUnavailable(msg) from exc


def hooke_spring_force_native(
    positions: Tensor,
    edges: Tensor,
    rest_lengths: float | Tensor,
    stiffness: float | Tensor,
    *,
    velocities: Tensor | None = None,
    damping: float | Tensor = 0.0,
) -> Tensor:
    """Evaluate Hooke spring forces through the optional C++ extension."""
    extension = _load_spring_extension()
    original_shape = positions.shape
    edge_shape = (*positions.shape[:-2], edges.shape[0])
    flat_positions = positions.reshape(-1, *positions.shape[-2:])
    flat_velocities = None if velocities is None else velocities.reshape_as(flat_positions)

    def edge_parameter(value: float | Tensor) -> Tensor:
        tensor = _as_tensor_like(value, positions)
        return torch.broadcast_to(tensor, edge_shape).reshape(-1, edges.shape[0])

    result = extension.hooke_spring_force(
        flat_positions,
        edges,
        edge_parameter(rest_lengths),
        edge_parameter(stiffness),
        flat_velocities,
        edge_parameter(damping),
    )
    return result.reshape(original_shape)


def pairwise_gravity_force_native(
    positions: Tensor,
    masses: float | Tensor,
    *,
    gravitational_constant: float = 1.0,
    softening: float = 0.0,
) -> Tensor:
    """Evaluate pairwise gravity through the optional C++ extension."""
    extension = _load_spring_extension()
    return extension.pairwise_gravity_force(
        positions,
        _as_tensor_like(masses, positions),
        float(gravitational_constant),
        float(softening),
    )


def gravity_neighbor_list_native(positions: Tensor, cutoff: float) -> Tensor:
    """Build a cutoff neighbor list through the optional C++ extension."""
    return _load_spring_extension().gravity_neighbor_list(positions, float(cutoff))


def native_kernels_status() -> tuple[bool, str | None]:
    """Return whether the optional native extension can be loaded."""
    try:
        _load_spring_extension()
    except NativeExtensionUnavailable as exc:
        return False, str(exc)
    return True, None


def native_kernels_available() -> bool:
    """Return ``True`` when the optional native extension can be loaded."""
    available, _ = native_kernels_status()
    return available


def native_spring_status() -> tuple[bool, str | None]:
    """Return whether the optional native extension can be loaded."""
    return native_kernels_status()


def native_spring_available() -> bool:
    """Return ``True`` when the optional native extension can be loaded."""
    return native_kernels_available()
