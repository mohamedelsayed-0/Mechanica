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
def _register_fake_kernels() -> None:
    def vector_shape(*args: Any) -> Tensor:
        return torch.empty_like(args[10])

    def matrix_shape(*args: Any) -> Tensor:
        q = args[10]
        return q.new_empty((*q.shape, q.shape[-1]))

    torch.library.register_fake("mechanica::rnea", vector_shape)
    torch.library.register_fake("mechanica::crba", matrix_shape)
    torch.library.register_fake("mechanica::aba", vector_shape)


@lru_cache(maxsize=1)
def _load_spring_extension() -> Any:
    try:
        extension = importlib.import_module("mechanica._mechanica_native")
        _register_fake_kernels()
        return extension
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - depends on local torch install
        raise NativeExtensionUnavailable("torch C++ extension loader is unavailable") from exc

    source_dir = Path(__file__).with_name("native")
    sources = [source_dir / "spring.cpp", source_dir / "robotics.cpp"]
    if any(not source.exists() for source in sources):
        raise NativeExtensionUnavailable(f"native source files are missing from {source_dir}")

    extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"]
    verbose = os.environ.get(_VERBOSE_ENV, "").strip().lower() in _TRUTHY
    build_directory = os.environ.get(_BUILD_DIR_ENV)
    if build_directory:
        Path(build_directory).mkdir(parents=True, exist_ok=True)

    try:
        extension = load(
            name="mechanica_native_springs",
            sources=[str(source) for source in sources],
            build_directory=build_directory,
            extra_cflags=extra_cflags,
            with_cuda=False,
            verbose=verbose,
        )
        _register_fake_kernels()
        return extension
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


def so3_exp_native(rotation_vector: Tensor) -> Tensor:
    """Evaluate the registered batched SO(3) exponential operator."""
    _load_spring_extension()
    return torch.ops.mechanica.so3_exp(rotation_vector)


def forward_kinematics_native(model: Any, q: Tensor) -> Tensor:
    """Evaluate registered batched rigid-tree forward kinematics."""
    _load_spring_extension()
    return torch.ops.mechanica.forward_kinematics(
        model.parents,
        model.joint_types,
        model.q_indices,
        model.axes,
        model.joint_origins,
        model.multipliers,
        model.offsets,
        q,
    )


def _robot_dynamics_arguments(model: Any) -> tuple[Tensor, ...]:
    return (
        model.parents,
        model.joint_types,
        model.q_indices,
        model.axes,
        model.joint_origins,
        model.multipliers,
        model.offsets,
        model.masses,
        model.centers_of_mass,
        model.inertias,
    )


def inverse_dynamics_rnea_native(
    model: Any,
    q: Tensor,
    qdot: Tensor,
    qddot: Tensor,
    gravity: Tensor,
    external_forces: Tensor | None = None,
) -> Tensor:
    """Evaluate batched RNEA through the registered native operator."""
    if not torch.compiler.is_compiling():
        _load_spring_extension()
    return torch.ops.mechanica.rnea(
        *_robot_dynamics_arguments(model), q, qdot, qddot, gravity, external_forces
    )


def mass_matrix_crba_native(model: Any, q: Tensor) -> Tensor:
    """Evaluate batched CRBA through the registered native operator."""
    if not torch.compiler.is_compiling():
        _load_spring_extension()
    return torch.ops.mechanica.crba(*_robot_dynamics_arguments(model), q)


def forward_dynamics_aba_native(
    model: Any,
    q: Tensor,
    qdot: Tensor,
    generalized_forces: Tensor,
    gravity: Tensor,
) -> Tensor:
    """Evaluate batched ABA through the registered native operator."""
    if not torch.compiler.is_compiling():
        _load_spring_extension()
    return torch.ops.mechanica.aba(
        *_robot_dynamics_arguments(model), q, qdot, generalized_forces, gravity
    )


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
