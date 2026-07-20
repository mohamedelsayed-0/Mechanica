import torch

from mechanica import forward_kinematics, inverse_dynamics_rnea, load_urdf, mass_matrix_crba

URDF = """
<robot name="pendulum">
  <link name="base"/>
  <link name="arm">
    <inertial><origin xyz="0.5 0 0"/><mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
</robot>
"""

model = load_urdf(URDF)
q = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
qdot = torch.zeros_like(q)
pose = forward_kinematics(model, q)[model.link_index("arm")]
center = pose[:3, :3] @ model.centers_of_mass[model.link_index("arm")] + pose[:3, 3]
mass = mass_matrix_crba(model, q)
gravity = inverse_dynamics_rnea(model, q, qdot, torch.zeros_like(q))

print("arm center", center.detach())
print("mass matrix", mass.detach())
print("gravity torque", gravity.detach())
