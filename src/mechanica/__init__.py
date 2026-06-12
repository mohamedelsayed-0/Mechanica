"""PyTorch-native analytical and classical mechanics."""

from .analytical import HamiltonianSystem, LagrangianSystem
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
from .fit import FitResult, fit_lagrangian_residual
from .integrators import euler_step, rk4_step, semi_implicit_euler_step, velocity_verlet_step
from .kinematics import estimate_acceleration, estimate_velocity

__all__ = [
    "DiagnosticsReport",
    "FitResult",
    "HamiltonianSystem",
    "LagrangianSystem",
    "angular_momentum",
    "center_of_mass",
    "energy_drift",
    "estimate_acceleration",
    "estimate_velocity",
    "euler_step",
    "fit_lagrangian_residual",
    "hooke_spring_force",
    "kinetic_energy",
    "lagrangian_diagnostics",
    "linear_momentum",
    "near_surface_gravity_force",
    "newton_residual",
    "residual_stats",
    "rk4_step",
    "semi_implicit_euler_step",
    "velocity_verlet_step",
]

__version__ = "0.1.0"
