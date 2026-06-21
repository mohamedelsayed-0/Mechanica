"""Planar two-link arm forward kinematics and Jacobian."""

import torch

from mechanica import (
    operational_space_velocity,
    planar_chain_points,
    planar_forward_kinematics,
    planar_jacobian,
)

q = torch.tensor([0.4, -0.8])
qdot = torch.tensor([0.5, 0.2])
link_lengths = torch.tensor([1.0, 0.8])

points = planar_chain_points(q, link_lengths)
end_effector = planar_forward_kinematics(q, link_lengths)
jacobian = planar_jacobian(q, link_lengths)
velocity = operational_space_velocity(jacobian, qdot)

print("joint/world points:")
print(points)
print("end effector:", end_effector)
print("jacobian:")
print(jacobian)
print("end-effector velocity:", velocity)
