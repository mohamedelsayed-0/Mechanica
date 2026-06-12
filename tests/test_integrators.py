import torch

from mechanica import rk4_step


def test_rk4_step_accepts_tuple_state() -> None:
    def dynamics(t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        q, v = state
        return v, -q

    q0 = torch.tensor([1.0])
    v0 = torch.tensor([0.0])
    q1, v1 = rk4_step(dynamics, torch.tensor(0.0), (q0, v0), 0.01)

    assert q1.shape == q0.shape
    assert v1.shape == v0.shape
    assert q1 < q0
