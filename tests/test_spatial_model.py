import torch

from mechanica.robot_model import REVOLUTE, load_urdf
from mechanica.spatial import se3_exp, se3_log, so3_exp, so3_log


URDF = """
<robot name="arm">
  <link name="base"/>
  <link name="tip">
    <inertial>
      <origin xyz="0.5 0 0"/>
      <mass value="2"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.3"/>
    </inertial>
    <collision><origin xyz="0.5 0 0"/><geometry><sphere radius="0.2"/></geometry></collision>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="tip"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
  </joint>
</robot>
"""


def test_so3_and_se3_round_trip_with_batches() -> None:
    rotation_vector = torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]], dtype=torch.float64)
    assert torch.allclose(so3_log(so3_exp(rotation_vector)), rotation_vector, atol=1e-10)

    twist = torch.tensor([0.4, -0.1, 0.2, 0.1, 0.2, -0.1], dtype=torch.float64)
    assert torch.allclose(se3_log(se3_exp(twist)), twist, atol=1e-10)


def test_load_urdf_packs_tree_and_inertia() -> None:
    model = load_urdf(URDF)

    assert model.link_names == ("base", "tip")
    assert model.dof == 1
    assert model.parents.tolist() == [-1, 0]
    assert model.joint_types.tolist() == [0, REVOLUTE]
    assert torch.allclose(model.centers_of_mass[1], torch.tensor([0.5, 0, 0], dtype=model.masses.dtype))
    assert torch.equal(model.limits[0], torch.tensor([-1, 1], dtype=model.limits.dtype))
    assert model.collision_links.tolist() == [1]
    assert torch.allclose(model.collision_parameters[0], torch.tensor([0.2, 0, 0], dtype=model.masses.dtype))
