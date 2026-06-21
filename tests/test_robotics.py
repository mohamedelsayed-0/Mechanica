import torch

from mechanica import (
    LagrangianSystem,
    bias_forces,
    computed_torque,
    forward_dynamics,
    gravity_forces,
    inverse_dynamics,
    lagrangian_state_dynamics,
    manipulator_terms,
    mass_matrix,
    velocity_forces,
)


def spring_mass_system() -> LagrangianSystem:
    mass = torch.tensor(2.0)
    stiffness = torch.tensor(8.0)

    def kinetic(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        return 0.5 * mass * (qdot * qdot).sum()

    def potential(q: torch.Tensor) -> torch.Tensor:
        return 0.5 * stiffness * (q * q).sum()

    return LagrangianSystem(kinetic=kinetic, potential=potential)


def test_manipulator_terms_for_spring_mass() -> None:
    system = spring_mass_system()
    q = torch.tensor([[0.5], [1.0]])
    qdot = torch.tensor([[0.0], [2.0]])

    terms = manipulator_terms(system, q, qdot)

    assert torch.allclose(terms.mass_matrix, torch.full((2, 1, 1), 2.0))
    assert torch.allclose(terms.bias_forces, 8.0 * q)


def test_robotics_convention_helpers_decompose_bias_terms() -> None:
    system = spring_mass_system()
    q = torch.tensor([[0.5], [1.0]])
    qdot = torch.tensor([[0.0], [2.0]])

    assert torch.allclose(mass_matrix(system, q), torch.full((2, 1, 1), 2.0))
    assert torch.allclose(bias_forces(system, q, qdot), 8.0 * q)
    assert torch.allclose(gravity_forces(system, q), 8.0 * q)
    assert torch.allclose(velocity_forces(system, q, qdot), torch.zeros_like(q))


def test_inverse_and_forward_dynamics_are_consistent() -> None:
    system = spring_mass_system()
    q = torch.tensor([[0.5], [1.0]])
    qdot = torch.tensor([[0.0], [2.0]])
    qddot = torch.tensor([[3.0], [-1.0]])

    tau = inverse_dynamics(system, q, qdot, qddot)
    recovered_qddot = forward_dynamics(system, q, qdot, tau)

    assert torch.allclose(tau, 2.0 * qddot + 8.0 * q)
    assert torch.allclose(recovered_qddot, qddot)


def test_forward_dynamics_maps_underactuated_input_matrix() -> None:
    def kinetic(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        del q
        return 0.5 * (qdot * qdot).sum()

    system = LagrangianSystem(kinetic=kinetic)
    q = torch.tensor([0.0, 0.0])
    qdot = torch.tensor([0.0, 0.0])
    input_matrix = torch.tensor([[0.0], [1.0]])

    qddot = forward_dynamics(
        system,
        q,
        qdot,
        torch.tensor([3.0]),
        input_matrix=input_matrix,
    )

    assert torch.allclose(qddot, torch.tensor([0.0, 3.0]))


def test_lagrangian_state_dynamics_accepts_generalized_forces() -> None:
    system = spring_mass_system()
    dynamics = lagrangian_state_dynamics(system)

    derivative = dynamics(torch.tensor(0.0), torch.tensor([1.0, 0.5]), torch.tensor([10.0]))

    assert torch.allclose(derivative, torch.tensor([0.5, 1.0]))


def test_lagrangian_state_dynamics_accepts_underactuated_controls() -> None:
    def kinetic(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        del q
        return 0.5 * (qdot * qdot).sum()

    system = LagrangianSystem(kinetic=kinetic)
    dynamics = lagrangian_state_dynamics(system, input_matrix=torch.tensor([[0.0], [1.0]]))

    derivative = dynamics(
        torch.tensor(0.0),
        torch.tensor([0.0, 0.0, 1.0, 2.0]),
        torch.tensor([4.0]),
    )

    assert torch.allclose(derivative, torch.tensor([1.0, 2.0, 0.0, 4.0]))


def test_computed_torque_adds_pd_acceleration_feedback() -> None:
    system = spring_mass_system()

    tau = computed_torque(
        system,
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        kp=2.0,
    )

    assert torch.allclose(tau, torch.tensor([4.0]))
