import os

import pytest
import torch

from mechanica import gravity_neighbor_list, hooke_spring_force
from mechanica.spatial import so3_exp
from mechanica.rigid_body import forward_kinematics
from mechanica.robot_model import load_urdf

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


def test_registered_native_so3_matches_torch_and_backpropagates() -> None:
    vector = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64, requires_grad=True)
    expected = so3_exp(vector, use_native=False)
    actual = so3_exp(vector, use_native=True)
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert vector.grad is not None and torch.isfinite(vector.grad).all()


def test_registered_native_forward_kinematics_matches_torch() -> None:
    model = load_urdf("""
    <robot name="arm"><link name="base"/><link name="tip"/>
      <joint name="joint" type="revolute"><parent link="base"/><child link="tip"/>
        <axis xyz="0 0 1"/></joint>
    </robot>
    """)
    q = torch.tensor([[0.1], [0.2]], dtype=torch.float64, requires_grad=True)
    expected = forward_kinematics(model, q, use_native=False)
    actual = forward_kinematics(model, q, use_native=True)
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
