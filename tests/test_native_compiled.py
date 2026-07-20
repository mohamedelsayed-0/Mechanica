import os

import pytest
import torch

from mechanica import gravity_neighbor_list, hooke_spring_force
from mechanica.spatial import so3_exp
from mechanica.rigid_body import (
    forward_dynamics_aba,
    forward_kinematics,
    inverse_dynamics_rnea,
    mass_matrix_crba,
)
from mechanica.robot_model import load_urdf

pytestmark = pytest.mark.skipif(
    os.environ.get("MECHANICA_TEST_NATIVE") != "1",
    reason="compiled native checks are opt-in",
)

DYNAMICS_URDF = """
<robot name="two_link">
  <link name="base"/>
  <link name="first"><inertial><origin xyz="0.5 0 0"/><mass value="2"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.4"/>
  </inertial></link>
  <link name="second"><inertial><origin xyz="0.4 0 0"/><mass value="1"/>
    <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.15" iyz="0" izz="0.2"/>
  </inertial></link>
  <joint name="first_joint" type="revolute"><parent link="base"/><child link="first"/>
    <axis xyz="0 0 1"/></joint>
  <joint name="second_joint" type="revolute"><parent link="first"/><child link="second"/>
    <origin xyz="1 0 0"/><axis xyz="0 1 0"/></joint>
</robot>
"""


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_rigid_body_algorithms_match_torch(dtype: torch.dtype) -> None:
    model = load_urdf(DYNAMICS_URDF, dtype=dtype)
    q = torch.tensor([[0.2, -0.3], [-0.1, 0.4]], dtype=dtype)
    qdot = torch.tensor([[0.3, 0.1], [-0.2, 0.5]], dtype=dtype)
    qddot = torch.tensor([[0.7, -0.4], [0.2, 0.6]], dtype=dtype)
    gravity = torch.tensor([0.5, -0.2, -9.7], dtype=dtype)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10

    expected_torque = inverse_dynamics_rnea(
        model, q, qdot, qddot, gravity=gravity, use_native=False
    )
    actual_torque = inverse_dynamics_rnea(
        model, q, qdot, qddot, gravity=gravity, use_native=True
    )
    expected_mass = mass_matrix_crba(model, q, use_native=False)
    actual_mass = mass_matrix_crba(model, q, use_native=True)
    actual_acceleration = forward_dynamics_aba(
        model, q, qdot, actual_torque, gravity=gravity, use_native=True
    )

    assert torch.allclose(actual_torque, expected_torque, atol=tolerance, rtol=tolerance)
    assert torch.allclose(actual_mass, expected_mass, atol=tolerance, rtol=tolerance)
    assert torch.allclose(actual_acceleration, qddot, atol=tolerance, rtol=tolerance)


def test_native_rigid_body_ops_support_gradients_layouts_and_empty_batches() -> None:
    model = load_urdf(DYNAMICS_URDF)
    storage = torch.tensor(
        [[0.2, 9.0, -0.3, 9.0], [-0.1, 9.0, 0.4, 9.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    q = storage[:, ::2]
    qdot = torch.tensor([[0.3, 0.1], [-0.2, 0.5]], dtype=torch.float64)
    qddot = torch.tensor([[0.7, -0.4], [0.2, 0.6]], dtype=torch.float64)
    gravity = torch.tensor([0.5, -0.2, -9.7], dtype=torch.float64)

    torque = inverse_dynamics_rnea(model, q, qdot, qddot, gravity=gravity, use_native=True)
    mass = mass_matrix_crba(model, q, use_native=True)
    first_gradient = torch.autograd.grad(torque.square().sum() + mass.square().sum(), storage,
                                         create_graph=True)[0]
    torch.autograd.grad(first_gradient.sum(), storage)
    assert torch.isfinite(first_gradient).all()

    empty = torch.empty(0, model.dof, dtype=torch.float64)
    assert inverse_dynamics_rnea(
        model, empty, empty, empty, gravity=gravity, use_native=True
    ).shape == empty.shape
    assert mass_matrix_crba(model, empty, use_native=True).shape == (0, model.dof, model.dof)


def test_native_rnea_runs_under_torch_compile() -> None:
    model = load_urdf(DYNAMICS_URDF)
    gravity = torch.tensor([0.0, 0.0, -9.80665], dtype=torch.float64)

    def evaluate(q, qdot, qddot):
        return inverse_dynamics_rnea(
            model, q, qdot, qddot, gravity=gravity, use_native=True
        )

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    q = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    qdot = torch.tensor([[0.3, 0.1]], dtype=torch.float64)
    qddot = torch.tensor([[0.7, -0.4]], dtype=torch.float64)
    assert torch.allclose(compiled(q, qdot, qddot), evaluate(q, qdot, qddot))
