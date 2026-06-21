import torch

from mechanica import (
    operational_space_velocity,
    planar_chain_points,
    planar_end_effector_pose,
    planar_forward_kinematics,
    planar_jacobian,
    planar_link_positions,
)


def test_planar_forward_kinematics_two_link_arm() -> None:
    q = torch.tensor([0.0, torch.pi / 2])
    lengths = torch.tensor([1.0, 2.0])

    position = planar_forward_kinematics(q, lengths)
    pose = planar_end_effector_pose(q, lengths)

    assert torch.allclose(position, torch.tensor([1.0, 2.0]), atol=1e-6)
    assert torch.allclose(pose, torch.tensor([1.0, 2.0, torch.pi / 2]), atol=1e-6)


def test_planar_link_positions_and_chain_points_support_batches() -> None:
    q = torch.tensor([[0.0, 0.0], [torch.pi / 2, 0.0]])
    lengths = torch.tensor([1.0, 1.0])

    positions = planar_link_positions(q, lengths)
    points = planar_chain_points(q, lengths)

    assert positions.shape == (2, 2, 2)
    assert points.shape == (2, 3, 2)
    assert torch.allclose(positions[0], torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
    assert torch.allclose(points[1, -1], torch.tensor([0.0, 2.0]), atol=1e-6)


def test_planar_jacobian_matches_autograd() -> None:
    q = torch.tensor([0.4, -0.7, 0.2], requires_grad=True)
    lengths = torch.tensor([0.5, 1.2, 0.8])

    jacobian = planar_jacobian(q, lengths)
    autograd_jacobian = torch.autograd.functional.jacobian(
        lambda q_value: planar_forward_kinematics(q_value, lengths),
        q,
    )

    assert jacobian.shape == (2, 3)
    assert torch.allclose(jacobian, autograd_jacobian, atol=1e-6)


def test_operational_space_velocity_uses_jacobian() -> None:
    q = torch.tensor([0.2, 0.3])
    qdot = torch.tensor([1.5, -0.5])
    lengths = torch.tensor([1.0, 0.75])

    jacobian = planar_jacobian(q, lengths)
    velocity = operational_space_velocity(jacobian, qdot)

    assert torch.allclose(velocity, jacobian @ qdot)
