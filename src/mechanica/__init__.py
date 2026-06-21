"""PyTorch-native analytical and classical mechanics."""

from ._native import NativeExtensionUnavailable
from .analytical import HamiltonianSystem, LagrangianSystem, canonical_transformation_residual
from .classical import (
    angular_momentum,
    center_of_mass,
    hooke_spring_force,
    kinetic_energy,
    linear_momentum,
    near_surface_gravity_force,
    newton_residual,
)
from .diagnostics import DiagnosticsReport, energy_drift, lagrangian_diagnostics, residual_stats
from .dynamics import (
    euler_state_step,
    hamiltonian_dynamics,
    join_state,
    linearize,
    rk4_state_step,
    rollout,
    second_order_dynamics,
    split_state,
)
from .fit import FitResult, fit_lagrangian_residual
from .integrators import euler_step, rk4_step, semi_implicit_euler_step, velocity_verlet_step
from .kinematics import estimate_acceleration, estimate_velocity

__all__ = [
    "DiagnosticsReport",
    "FitResult",
    "HamiltonianSystem",
    "LagrangianSystem",
    "NativeExtensionUnavailable",
    "angular_momentum",
    "canonical_transformation_residual",
    "center_of_mass",
    "energy_drift",
    "estimate_acceleration",
    "estimate_velocity",
    "euler_step",
    "euler_state_step",
    "fit_lagrangian_residual",
    "hamiltonian_dynamics",
    "hooke_spring_force",
    "join_state",
    "kinetic_energy",
    "lagrangian_diagnostics",
    "linear_momentum",
    "linearize",
    "near_surface_gravity_force",
    "newton_residual",
    "residual_stats",
    "rk4_step",
    "rk4_state_step",
    "rollout",
    "semi_implicit_euler_step",
    "second_order_dynamics",
    "split_state",
    "velocity_verlet_step",
]

__version__ = "0.1.0"
