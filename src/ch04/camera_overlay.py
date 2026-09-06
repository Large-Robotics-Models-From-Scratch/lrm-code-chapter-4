"""Approximate SO-101 forward-kinematics overlays for open-loop analysis.

The dataset contains calibrated joint positions but not camera intrinsics or
extrinsics.  This module therefore keeps the two approximations separate:
URDF forward kinematics reconstructs a metric end-effector path, while a
weak-perspective affine map projects that path into one fixed camera view.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SO101_ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
SO101_URDF_URL = (
    "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/"
    "main/Simulation/SO101/so101_new_calib.urdf"
)


def _numbers(value: str | None, length: int = 3) -> np.ndarray:
    if value is None:
        return np.zeros(length, dtype=np.float64)
    parsed = np.fromstring(value, sep=" ", dtype=np.float64)
    if parsed.shape != (length,):
        raise ValueError(f"expected {length} numbers, received {value!r}")
    return parsed


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        return np.eye(3)
    x, y, z = axis / norm
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (1 - np.cos(angle)) * (skew @ skew)
    )


def _transform(xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    kind: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class UrdfChain:
    """One serial URDF chain from a base link to a tip link."""

    joints: tuple[UrdfJoint, ...]
    base_link: str
    tip_link: str

    @classmethod
    def from_file(
        cls,
        urdf_path: str | Path,
        tip_link: str = "gripper_frame_link",
    ) -> "UrdfChain":
        root = ET.parse(urdf_path).getroot()
        by_child: dict[str, UrdfJoint] = {}
        for element in root.findall("joint"):
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                continue
            origin = element.find("origin")
            axis = element.find("axis")
            joint = UrdfJoint(
                name=element.attrib["name"],
                kind=element.attrib.get("type", "fixed"),
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                xyz=_numbers(
                    None if origin is None else origin.attrib.get("xyz")
                ),
                rpy=_numbers(
                    None if origin is None else origin.attrib.get("rpy")
                ),
                axis=_numbers(
                    None if axis is None else axis.attrib.get("xyz")
                ),
            )
            by_child[joint.child] = joint

        reversed_chain = []
        link = tip_link
        while link in by_child:
            joint = by_child[link]
            reversed_chain.append(joint)
            link = joint.parent
        if not reversed_chain:
            raise ValueError(
                f"tip link {tip_link!r} is not connected in the URDF"
            )
        return cls(tuple(reversed(reversed_chain)), link, tip_link)

    def link_positions(
        self,
        joint_positions_deg: np.ndarray,
        joint_names: tuple[str, ...] = SO101_ARM_JOINT_NAMES,
    ) -> dict[str, np.ndarray]:
        values = np.asarray(joint_positions_deg, dtype=np.float64).reshape(
            -1
        )
        if values.size < len(joint_names):
            raise ValueError(
                "joint_positions_deg does not contain every arm joint"
            )
        angles = dict(
            zip(joint_names, np.deg2rad(values[: len(joint_names)]))
        )
        pose = np.eye(4, dtype=np.float64)
        positions = {self.base_link: pose[:3, 3].copy()}
        for joint in self.joints:
            pose = pose @ _transform(joint.xyz, _rpy_matrix(joint.rpy))
            if joint.kind in {"revolute", "continuous"}:
                pose = pose @ _transform(
                    np.zeros(3),
                    _axis_angle_matrix(
                        joint.axis, angles.get(joint.name, 0.0)
                    ),
                )
            elif joint.kind == "prismatic":
                pose = pose @ _transform(
                    joint.axis * angles.get(joint.name, 0.0), np.eye(3)
                )
            positions[joint.child] = pose[:3, 3].copy()
        return positions

    def end_effector(self, joint_positions_deg: np.ndarray) -> np.ndarray:
        return self.link_positions(joint_positions_deg)[self.tip_link]


def apply_position_targets(
    initial_state: np.ndarray,
    predicted_actions: np.ndarray,
    tracking_alpha: float = 1.0,
) -> np.ndarray:
    """Apply absolute joint-position targets to a simple first-order servo.

    With ``tracking_alpha=1``, each action becomes the next realized joint
    state. Smaller values visualize lag without pretending to simulate
    dynamics.
    """
    state = np.asarray(initial_state, dtype=np.float64).reshape(-1)
    targets = np.asarray(predicted_actions, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != state.size:
        raise ValueError("predicted_actions must have shape [H, state_dim]")
    if not 0 < tracking_alpha <= 1:
        raise ValueError("tracking_alpha must lie in (0, 1]")
    trajectory = [state.copy()]
    for target in targets:
        state = state + tracking_alpha * (target - state)
        trajectory.append(state.copy())
    return np.stack(trajectory)


def end_effector_path(
    chain: UrdfChain,
    initial_state: np.ndarray,
    predicted_actions: np.ndarray,
    tracking_alpha: float = 1.0,
) -> np.ndarray:
    states = apply_position_targets(
        initial_state, predicted_actions, tracking_alpha
    )
    return np.stack([chain.end_effector(state) for state in states])


def fit_affine_camera(
    world_points: np.ndarray,
    image_points: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit a weak-perspective 3-D-to-pixel affine map and return RMSE."""
    world = np.asarray(world_points, dtype=np.float64)
    pixels = np.asarray(image_points, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 3:
        raise ValueError("world_points must have shape [N, 3]")
    if pixels.shape != (world.shape[0], 2) or world.shape[0] < 4:
        raise ValueError(
            "image_points must match at least four world points"
        )
    design = np.column_stack([world, np.ones(world.shape[0])])
    projection, _, _, _ = np.linalg.lstsq(design, pixels, rcond=None)
    projected = design @ projection
    rmse = float(np.sqrt(np.mean(np.square(projected - pixels))))
    return projection, rmse


def project_to_image(
    points: np.ndarray, projection: np.ndarray
) -> np.ndarray:
    world = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(projection, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if matrix.shape != (4, 2):
        raise ValueError("projection must have shape [4, 2]")
    return np.column_stack([world, np.ones(world.shape[0])]) @ matrix


def plot_camera_trajectory_overlay(
    image: np.ndarray,
    world_path: np.ndarray,
    projection: np.ndarray,
    *,
    calibration_rmse_px: float | None = None,
):
    """Draw an approximate FK path over a camera frame."""
    import matplotlib.pyplot as plt

    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[-1] not in {3, 4}:
        raise ValueError("image must have shape [height, width, channels]")
    pixels = project_to_image(world_path, projection)
    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    axis.imshow(frame)
    axis.plot(
        pixels[:, 0],
        pixels[:, 1],
        color="#FF7F0E",
        linewidth=2.2,
        marker="o",
        markersize=3.4,
        label="predicted end-effector path",
    )
    axis.scatter(
        pixels[0, 0],
        pixels[0, 1],
        s=58,
        color="#1F77B4",
        edgecolor="white",
        linewidth=0.9,
        label="current proprioceptive state",
        zorder=4,
    )
    axis.scatter(
        pixels[-1, 0],
        pixels[-1, 1],
        s=64,
        marker="X",
        color="#2CA02C",
        edgecolor="white",
        linewidth=0.8,
        label="final predicted target",
        zorder=4,
    )
    axis.set_axis_off()
    axis.legend(loc="lower right", frameon=True, fontsize=8)
    note = "Approximate FK projection; not a calibrated camera measurement"
    if calibration_rmse_px is not None:
        note += f" (anchor RMSE {calibration_rmse_px:.1f} px)"
    axis.text(
        0.01,
        0.99,
        note,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox={
            "facecolor": "#24292F",
            "alpha": 0.78,
            "edgecolor": "none",
            "pad": 4,
        },
    )
    figure.tight_layout(pad=0)
    return figure
