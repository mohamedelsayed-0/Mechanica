import torch

from mechanica.rigid_body import (
    forward_dynamics_aba,
    forward_kinematics,
    geometric_jacobian,
    inverse_dynamics_rnea,
    inverse_kinematics,
    mass_matrix_crba,
)
from mechanica.robot_model import load_urdf


URDF = """
<robot name="arm">
  <link name="base"/>
  <link name="arm">
    <inertial><origin xyz="0.5 0 0"/><mass value="2"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.3"/>
    </inertial>
  </link>
  <link name="tool"/>
  <joint name="joint" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
  </joint>
  <joint name="tool_fixed" type="fixed">
    <parent link="arm"/><child link="tool"/><origin xyz="1 0 0"/>
  </joint>
</robot>
"""


def test_tree_forward_kinematics_and_geometric_jacobian() -> None:
    model = load_urdf(URDF)
    q = torch.tensor([0.0], dtype=torch.float64)

    pose = forward_kinematics(model, q)[model.link_index("tool")]
    jacobian = geometric_jacobian(model, q, "tool")

    assert torch.allclose(pose[:3, 3], torch.tensor([1.0, 0.0, 0.0], dtype=q.dtype))
    assert torch.allclose(jacobian[3:, 0], torch.tensor([0.0, 1.0, 0.0], dtype=q.dtype))


def test_rnea_crba_and_aba_are_consistent() -> None:
    model = load_urdf(URDF)
    q = torch.tensor([0.2], dtype=torch.float64)
    qdot = torch.tensor([0.3], dtype=torch.float64)
    qddot = torch.tensor([2.0], dtype=torch.float64)
    gravity = torch.zeros(3, dtype=q.dtype)

    mass = mass_matrix_crba(model, q)
    torque = inverse_dynamics_rnea(model, q, qdot, qddot, gravity=gravity)
    bias = inverse_dynamics_rnea(model, q, qdot, torch.zeros_like(q), gravity=gravity)
    recovered = forward_dynamics_aba(model, q, qdot, torque, gravity=gravity)

    assert torch.allclose(torque, mass @ qddot + bias, atol=1e-10)
    assert torch.allclose(recovered, qddot, atol=1e-10)

    batched = inverse_dynamics_rnea(
        model, torch.stack((q, q)), torch.stack((qdot, qdot)),
        torch.stack((qddot, qddot)), gravity=gravity,
    )
    assert torch.allclose(batched, torque.expand(2, 1))


def test_inverse_kinematics_reaches_planar_target() -> None:
    model = load_urdf(URDF)
    q = inverse_kinematics(
        model,
        torch.zeros(1, dtype=torch.float64),
        "tool",
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
    )
    position = forward_kinematics(model, q)[model.link_index("tool"), :3, 3]
    assert torch.allclose(position, torch.tensor([0.0, 1.0, 0.0], dtype=q.dtype), atol=1e-5)
