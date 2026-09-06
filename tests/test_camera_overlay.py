from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ch04.camera_overlay import (
    UrdfChain,
    apply_position_targets,
    end_effector_path,
    fit_affine_camera,
    plot_camera_trajectory_overlay,
    project_to_image,
)

MINIMAL_URDF = """\
<robot name="test">
  <link name="base"/>
  <link name="arm"/>
  <link name="gripper_frame_link"/>
  <joint name="shoulder_pan" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="tip" type="fixed">
    <parent link="arm"/><child link="gripper_frame_link"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


def test_position_targets_start_from_proprioception():
    states = apply_position_targets(
        np.array([0.0, 10.0]),
        np.array([[4.0, 6.0], [8.0, 2.0]]),
        tracking_alpha=0.5,
    )
    np.testing.assert_allclose(states, [[0, 10], [2, 8], [5, 5]])
    with pytest.raises(ValueError, match="tracking_alpha"):
        apply_position_targets(np.zeros(2), np.zeros((1, 2)), 0.0)


def test_urdf_fk_and_camera_projection(tmp_path: Path):
    urdf = tmp_path / "test.urdf"
    urdf.write_text(MINIMAL_URDF)
    chain = UrdfChain.from_file(urdf)
    path = end_effector_path(
        chain,
        np.zeros(5),
        np.array([[90, 0, 0, 0, 0]], dtype=float),
    )
    np.testing.assert_allclose(path[0], [2, 0, 0], atol=1e-6)
    np.testing.assert_allclose(path[1], [1, 1, 0], atol=1e-6)

    world = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        dtype=float,
    )
    expected = np.column_stack(
        [2 * world[:, 0] + 10, -3 * world[:, 2] + 20]
    )
    projection, rmse = fit_affine_camera(world, expected)
    np.testing.assert_allclose(
        project_to_image(world, projection), expected
    )
    assert rmse < 1e-10

    figure = plot_camera_trajectory_overlay(
        np.zeros((40, 60, 3), dtype=np.uint8),
        world[:2],
        projection,
        calibration_rmse_px=rmse,
    )
    labels = figure.axes[0].get_legend_handles_labels()[1]
    assert "predicted end-effector path" in labels
    assert "current proprioceptive state" in labels
    plt.close(figure)
