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
from .control import LQRResult, finite_horizon_lqr, quadratic_cost, tvlqr
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
from .robotics import (
    ManipulatorTerms,
    computed_torque,
    forward_dynamics,
    inverse_dynamics,
    lagrangian_state_dynamics,
    manipulator_terms,
)

__all__ = [
    "DiagnosticsReport",
    "FitResult",
    "HamiltonianSystem",
    "LQRResult",
    "LagrangianSystem",
    "ManipulatorTerms",
    "NativeExtensionUnavailable",
    "angular_momentum",
    "canonical_transformation_residual",
    "center_of_mass",
    "computed_torque",
    "energy_drift",
    "estimate_acceleration",
    "estimate_velocity",
    "euler_step",
    "euler_state_step",
    "finite_horizon_lqr",
    "fit_lagrangian_residual",
    "forward_dynamics",
    "hamiltonian_dynamics",
    "hooke_spring_force",
    "inverse_dynamics",
    "join_state",
    "kinetic_energy",
    "lagrangian_diagnostics",
    "lagrangian_state_dynamics",
    "linear_momentum",
    "linearize",
    "manipulator_terms",
    "near_surface_gravity_force",
    "newton_residual",
    "quadratic_cost",
    "residual_stats",
    "rk4_step",
    "rk4_state_step",
    "rollout",
    "semi_implicit_euler_step",
    "second_order_dynamics",
    "split_state",
    "tvlqr",
    "velocity_verlet_step",
]

__version__ = "0.1.0"
