import os

import pytest
import torch

from mechanica import gravity_neighbor_list, hooke_spring_force

pytestmark = pytest.mark.skipif(
    os.environ.get("MECHANICA_TEST_NATIVE") != "1",
    reason="compiled native checks are opt-in",
)


def test_native_batched_springs_match_torch_and_backpropagate() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [3.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edges = torch.tensor([[0, 1]])
    expected = hooke_spring_force(positions, edges, 1.0, 2.0, use_native=False)
    actual = hooke_spring_force(positions, edges, 1.0, 2.0, use_native=True)

    assert torch.allclose(actual, expected)
    actual.square().sum().backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()


def test_native_neighbor_list_matches_torch() -> None:
    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    expected = gravity_neighbor_list(positions, 1.5, use_native=False)
    actual = gravity_neighbor_list(positions, 1.5, use_native=True)
    assert torch.equal(actual, expected)
