import torch

from mechanica import HamiltonianSystem, LagrangianSystem


def test_lagrangian_harmonic_oscillator_residual_is_small() -> None:
    m = torch.tensor(2.0)
    k = torch.tensor(8.0)

    def kinetic(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        return 0.5 * m * (qdot * qdot).sum()

    def potential(q: torch.Tensor) -> torch.Tensor:
        return 0.5 * k * (q * q).sum()

    system = LagrangianSystem(kinetic=kinetic, potential=potential)
    t = torch.linspace(0, 2, 32)
    w = torch.sqrt(k / m)
    q = torch.cos(w * t).unsqueeze(-1)
    qdot = -w * torch.sin(w * t).unsqueeze(-1)
    qddot = -(w**2) * torch.cos(w * t).unsqueeze(-1)

    residual = system.euler_lagrange_residual(q, qdot, qddot)

    assert residual.abs().max() < 1e-5


def test_hamiltonian_harmonic_oscillator_vector_field() -> None:
    m = torch.tensor(2.0)
    k = torch.tensor(8.0)

    def hamiltonian(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return 0.5 * (p * p).sum() / m + 0.5 * k * (q * q).sum()

    system = HamiltonianSystem(hamiltonian)
    q = torch.tensor([[0.5], [1.0]])
    p = torch.tensor([[2.0], [-1.0]])

    dqdt, dpdt = system.vector_field(q, p)

    assert torch.allclose(dqdt, p / m)
    assert torch.allclose(dpdt, -k * q)
