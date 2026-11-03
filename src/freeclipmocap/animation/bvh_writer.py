import io
from pathlib import Path

import numpy as np

from ..solver.rig_rest import RigRest

_BLENDER_TO_Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_POSITION_CHANNELS = "Xposition Yposition Zposition"
_ROTATION_CHANNELS = "Zrotation Xrotation Yrotation"


def write_bvh(
    path: Path,
    rig: RigRest,
    pose_rotations: np.ndarray,
    pose_heads: np.ndarray,
    frame_time_seconds: float,
) -> None:
    text = _hierarchy(rig) + _motion(rig, pose_rotations, pose_heads, frame_time_seconds)
    path.write_text(text, encoding="utf-8", newline="\n")


def _joint_name(bone_name: str) -> str:
    return "_".join(bone_name.split())


def _children_of(rig: RigRest) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in rig.bone_names]
    for index, parent in enumerate(rig.parent_indices):
        if parent >= 0:
            children[parent].append(index)
    return children


def _numbers(values: np.ndarray) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def _hierarchy(rig: RigRest) -> str:
    children = _children_of(rig)
    lines = ["HIERARCHY"]

    def write_joint(index: int, depth: int) -> None:
        indent = "\t" * depth
        parent = rig.parent_indices[index]
        if parent < 0:
            lines.append(f"{indent}ROOT {_joint_name(rig.bone_names[index])}")
            offset = rig.heads[index]
            channels = f"6 {_POSITION_CHANNELS} {_ROTATION_CHANNELS}"
        else:
            lines.append(f"{indent}JOINT {_joint_name(rig.bone_names[index])}")
            offset = rig.heads[index] - rig.heads[parent]
            channels = f"3 {_ROTATION_CHANNELS}"
        lines.append(f"{indent}{{")
        lines.append(f"{indent}\tOFFSET {_numbers(_BLENDER_TO_Y_UP @ offset)}")
        lines.append(f"{indent}\tCHANNELS {channels}")
        for child in children[index]:
            write_joint(child, depth + 1)
        if not children[index]:
            end_offset = _BLENDER_TO_Y_UP @ (rig.tails[index] - rig.heads[index])
            lines.extend((f"{indent}\tEnd Site", f"{indent}\t{{", f"{indent}\t\tOFFSET {_numbers(end_offset)}", f"{indent}\t}}"))
        lines.append(f"{indent}}}")

    for index, parent in enumerate(rig.parent_indices):
        if parent < 0:
            write_joint(index, 0)
    return "\n".join(lines) + "\n"


def _zxy_euler_degrees(rotations: np.ndarray) -> np.ndarray:
    z = np.arctan2(-rotations[..., 0, 1], rotations[..., 1, 1])
    x = np.arcsin(np.clip(rotations[..., 2, 1], -1.0, 1.0))
    y = np.arctan2(-rotations[..., 2, 0], rotations[..., 2, 2])
    return np.degrees(np.unwrap(np.stack((z, x, y), axis=-1), axis=0))


def _motion(rig: RigRest, pose_rotations: np.ndarray, pose_heads: np.ndarray, frame_time_seconds: float) -> str:
    changes_from_rest = pose_rotations @ np.swapaxes(rig.rest_rotations, -1, -2)
    parents = np.array(rig.parent_indices)
    is_root = parents < 0
    parent_changes = np.where(is_root[None, :, None, None], np.eye(3), changes_from_rest[:, np.maximum(parents, 0)])
    local_rotations = np.swapaxes(parent_changes, -1, -2) @ changes_from_rest
    angles = _zxy_euler_degrees(_BLENDER_TO_Y_UP @ local_rotations @ _BLENDER_TO_Y_UP.T)
    root_positions = pose_heads @ _BLENDER_TO_Y_UP.T
    columns = []
    for index in _hierarchy_order(rig):
        if is_root[index]:
            columns.append(root_positions[:, index])
        columns.append(angles[:, index])
    values = np.concatenate(columns, axis=1)
    buffer = io.StringIO()
    np.savetxt(buffer, values, fmt="%.6f", delimiter=" ")
    return f"MOTION\nFrames: {len(values)}\nFrame Time: {frame_time_seconds:.6f}\n{buffer.getvalue()}"


def _hierarchy_order(rig: RigRest) -> list[int]:
    children = _children_of(rig)
    order: list[int] = []

    def visit(index: int) -> None:
        order.append(index)
        for child in children[index]:
            visit(child)

    for index, parent in enumerate(rig.parent_indices):
        if parent < 0:
            visit(index)
    return order
