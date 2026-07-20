"""Compact tensor robot models and URDF loading."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from .spatial import transform

Tensor = torch.Tensor
FIXED, REVOLUTE, PRISMATIC = 0, 1, 2


def _numbers(text: str | None, default: tuple[float, ...]) -> list[float]:
    return list(default if text is None else map(float, text.split()))


def _origin(element: ET.Element | None, *, dtype: torch.dtype) -> Tensor:
    if element is None:
        return torch.eye(4, dtype=dtype)
    xyz = torch.tensor(_numbers(element.get("xyz"), (0, 0, 0)), dtype=dtype)
    roll, pitch, yaw = _numbers(element.get("rpy"), (0, 0, 0))
    cr, sr = torch.cos(torch.tensor(roll)), torch.sin(torch.tensor(roll))
    cp, sp = torch.cos(torch.tensor(pitch)), torch.sin(torch.tensor(pitch))
    cy, sy = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    rotation = torch.tensor(
        [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
         [-sp, cp * sr, cp * cr]],
        dtype=dtype,
    )
    return transform(rotation, xyz)


@dataclass(frozen=True)
class RobotModel:
    """A topologically ordered rigid-body tree packed into Torch tensors."""

    link_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    parents: Tensor
    joint_types: Tensor
    q_indices: Tensor
    axes: Tensor
    joint_origins: Tensor
    multipliers: Tensor
    offsets: Tensor
    masses: Tensor
    centers_of_mass: Tensor
    inertias: Tensor
    limits: Tensor

    @property
    def dof(self) -> int:
        return len(self.coordinate_names)

    @property
    def links(self) -> int:
        return len(self.link_names)

    def to(self, *args, **kwargs) -> RobotModel:
        fields = {
            name: getattr(self, name).to(*args, **kwargs)
            for name in (
                "parents", "joint_types", "q_indices", "axes", "joint_origins",
                "multipliers", "offsets", "masses", "centers_of_mass", "inertias", "limits"
            )
        }
        return replace(self, **fields)

    def link_index(self, name: str) -> int:
        return self.link_names.index(name)


def load_urdf(source: str | Path, *, dtype: torch.dtype = torch.float64) -> RobotModel:
    """Load a URDF file or XML string into a differentiable tensor model."""
    text = str(source)
    if isinstance(source, Path) or "<robot" not in text:
        text = Path(source).read_text()
    root = ET.fromstring(text)
    links = {element.get("name"): element for element in root.findall("link")}
    joints = {element.find("child").get("link"): element for element in root.findall("joint")}
    root_names = [name for name in links if name not in joints]
    if len(root_names) != 1:
        raise ValueError("URDF must contain one connected root link")

    children: dict[str, list[str]] = {}
    for child, joint in joints.items():
        children.setdefault(joint.find("parent").get("link"), []).append(child)
    order: list[str] = []
    stack = root_names[:]
    while stack:
        name = stack.pop(0)
        order.append(name)
        stack[0:0] = children.get(name, [])
    if len(order) != len(links):
        raise ValueError("URDF links must form one acyclic tree")

    joint_by_name = {joint.get("name"): joint for joint in joints.values()}
    coordinates = [
        joint.get("name") for joint in joints.values()
        if joint.get("type") != "fixed" and joint.find("mimic") is None
    ]
    coordinate_index = {name: index for index, name in enumerate(coordinates)}
    index = {name: position for position, name in enumerate(order)}
    parents, types, q_indices, axes, origins, multipliers, offsets = [], [], [], [], [], [], []
    masses, centers, inertias, joint_names = [], [], [], []

    for name in order:
        link = links[name]
        joint = joints.get(name)
        if joint is None:
            parents.append(-1)
            types.append(FIXED)
            q_indices.append(-1)
            axes.append([0, 0, 0])
            origins.append(torch.eye(4, dtype=dtype))
            multipliers.append(1.0)
            offsets.append(0.0)
            joint_names.append("")
        else:
            joint_type = joint.get("type")
            parents.append(index[joint.find("parent").get("link")])
            types.append(PRISMATIC if joint_type == "prismatic" else REVOLUTE if joint_type != "fixed" else FIXED)
            axis = joint.find("axis")
            axes.append(_numbers(None if axis is None else axis.get("xyz"), (1, 0, 0)))
            origins.append(_origin(joint.find("origin"), dtype=dtype))
            mimic = joint.find("mimic")
            source_name = joint.get("name") if mimic is None else mimic.get("joint")
            if source_name not in coordinate_index and joint_type != "fixed":
                source = joint_by_name.get(source_name)
                source_mimic = None if source is None else source.find("mimic")
                source_name = source_name if source_mimic is None else source_mimic.get("joint")
            q_indices.append(-1 if joint_type == "fixed" else coordinate_index[source_name])
            multipliers.append(1.0 if mimic is None else float(mimic.get("multiplier", "1")))
            offsets.append(0.0 if mimic is None else float(mimic.get("offset", "0")))
            joint_names.append(joint.get("name"))

        inertial = link.find("inertial")
        if inertial is None:
            masses.append(0.0)
            centers.append([0, 0, 0])
            inertias.append(torch.zeros(3, 3, dtype=dtype))
            continue
        mass = float(inertial.find("mass").get("value"))
        inertial_origin = _origin(inertial.find("origin"), dtype=dtype)
        values = inertial.find("inertia")
        matrix = torch.tensor(
            [[float(values.get("ixx")), float(values.get("ixy")), float(values.get("ixz"))],
             [float(values.get("ixy")), float(values.get("iyy")), float(values.get("iyz"))],
             [float(values.get("ixz")), float(values.get("iyz")), float(values.get("izz"))]],
            dtype=dtype,
        )
        masses.append(mass)
        centers.append(inertial_origin[:3, 3].tolist())
        rotation = inertial_origin[:3, :3]
        inertias.append(rotation @ matrix @ rotation.T)

    limits = torch.full((len(coordinates), 2), torch.nan, dtype=dtype)
    for joint in joints.values():
        name = joint.get("name")
        if name in coordinate_index and joint.find("limit") is not None:
            limit = joint.find("limit")
            limits[coordinate_index[name]] = torch.tensor(
                [float(limit.get("lower", "-inf")), float(limit.get("upper", "inf"))], dtype=dtype
            )
    return RobotModel(
        tuple(order), tuple(joint_names), tuple(coordinates), torch.tensor(parents),
        torch.tensor(types), torch.tensor(q_indices), torch.tensor(axes, dtype=dtype),
        torch.stack(origins), torch.tensor(multipliers, dtype=dtype),
        torch.tensor(offsets, dtype=dtype), torch.tensor(masses, dtype=dtype),
        torch.tensor(centers, dtype=dtype), torch.stack(inertias), limits,
    )
