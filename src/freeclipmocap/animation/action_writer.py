from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import bpy
import numpy as np
from mathutils import Euler, Quaternion

from ..rig.armature_rest import rig_rest_from_armature
from ..solver import rotations
from ..solver.pose_solver import SolvedTake
from ..solver.rig_rest import RigRest
from .key_reduction import (
    angle_between_rotations,
    select_key_frames,
    select_position_key_frames,
    worst_frames_of_each_run,
)

_MAXIMUM_REFINEMENT_PASSES = 10
_LOCATION_TOLERANCE_SHARE_OF_HEIGHT = 0.002
_MAXIMUM_MATCH_PASSES = 12
_MATCH_TOLERANCE_DEGREES = 0.1
_STALLED_SHARE = 0.01
_FOOT_DRIFT_SHARE_OF_LEVER = 0.002


@dataclass(frozen=True, slots=True)
class WrittenTake:
    action: bpy.types.Action
    unmatched_bones: tuple[str, ...]


def write_take(
    armature_object: bpy.types.Object,
    take: SolvedTake,
    action_name: str,
    start_frame: int,
    key_tolerance_degrees: float | None,
) -> WrittenTake:
    action = bpy.data.actions.new(action_name)
    action.use_fake_user = True
    animation_data = armature_object.animation_data or armature_object.animation_data_create()
    animation_data.action = action
    frames = start_frame + np.arange(take.frame_count, dtype=np.float64)
    for bone_name, locations in take.locations_by_bone.items():
        bone_path = f'pose.bones["{bpy.utils.escape_identifier(bone_name)}"]'
        fcurves = [
            action.fcurve_ensure_for_datablock(armature_object, f"{bone_path}.location", index=axis, group_name=bone_name)
            for axis in range(3)
        ]
        if key_tolerance_degrees is None:
            set_keys(fcurves, frames, locations, np.arange(take.frame_count))
        else:
            tolerance = _LOCATION_TOLERANCE_SHARE_OF_HEIGHT * armature_object.data.bones[bone_name].head_local.length
            write_reduced_keys(
                fcurves,
                frames,
                locations,
                select_position_key_frames(locations, tolerance),
                lambda evaluated: np.linalg.norm(evaluated - locations, axis=-1),
                tolerance,
            )
    quaternions_by_bone, unmatched_bones = _match_rig_constraints(armature_object, action, frames, take.rotations_by_bone)
    tolerances_by_bone = (
        key_tolerances(rig_rest_from_armature(armature_object), list(quaternions_by_bone), take.foot_bones, key_tolerance_degrees)
        if key_tolerance_degrees is not None
        else {}
    )
    for bone_name, quaternions in quaternions_by_bone.items():
        write_bone_rotations(armature_object, action, bone_name, frames, quaternions, tolerances_by_bone.get(bone_name))
    return WrittenTake(action=action, unmatched_bones=unmatched_bones)


def write_bone_rotations(
    armature_object: bpy.types.Object,
    action: bpy.types.Action,
    bone_name: str,
    frames: np.ndarray,
    quaternions: np.ndarray,
    tolerance_degrees: float | None,
) -> None:
    pose_bone = armature_object.pose.bones[bone_name]
    data_path, values = _channel_values(pose_bone, quaternions)
    fcurves = _rotation_fcurves(action, armature_object, bone_name, data_path, values.shape[1])
    if tolerance_degrees is None:
        set_keys(fcurves, frames, values, np.arange(len(frames)))
        return
    rotation_mode = pose_bone.rotation_mode
    write_reduced_keys(
        fcurves,
        frames,
        values,
        select_key_frames(quaternions, tolerance_degrees),
        lambda evaluated: angle_between_rotations(_as_quaternions(evaluated, rotation_mode), quaternions),
        np.radians(tolerance_degrees),
    )


def key_tolerances(
    rig: RigRest, bone_names: list[str], foot_bones: tuple[str, ...], tolerance_degrees: float
) -> dict[str, float]:
    levers: dict[str, float] = {}
    for foot_bone in foot_bones:
        foot_index = rig.index_of(foot_bone)
        index = foot_index
        while index >= 0:
            lever = float(np.linalg.norm(rig.tails[foot_index] - rig.heads[index]))
            levers[rig.bone_names[index]] = max(levers.get(rig.bone_names[index], 0.0), lever)
            index = rig.parent_indices[index]
    budget = _FOOT_DRIFT_SHARE_OF_LEVER * max(levers.values(), default=0.0)
    return {
        bone_name: min(tolerance_degrees, float(np.degrees(np.arctan2(budget, levers[bone_name]))))
        if levers.get(bone_name, 0.0) > 0.0
        else tolerance_degrees
        for bone_name in bone_names
    }


def _match_rig_constraints(
    armature_object: bpy.types.Object,
    action: bpy.types.Action,
    frames: np.ndarray,
    rotations_by_bone: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    if not rotations_by_bone or armature_object.name not in bpy.context.view_layer.objects:
        return dict(rotations_by_bone), ()
    local_rotations = {name: rotations.matrices_from_quaternions(quaternions) for name, quaternions in rotations_by_bone.items()}
    intended = _intended_rotations(rig_rest_from_armature(armature_object), local_rotations, len(frames))
    for bone_name, local in local_rotations.items():
        _write_every_frame(action, armature_object, bone_name, frames, local)
    tolerance = np.radians(_MATCH_TOLERANCE_DEGREES)
    unmatched: list[str] = []
    previous_worst = np.inf
    for match_pass in range(_MAXIMUM_MATCH_PASSES):
        evaluated_by_bone = _evaluated_rotations(armature_object, list(local_rotations), frames)
        errors = {bone_name: _largest_angle(evaluated, intended[bone_name]) for bone_name, evaluated in evaluated_by_bone.items()}
        unmatched = [bone_name for bone_name, error in errors.items() if error > tolerance]
        worst = max((errors[bone_name] for bone_name in unmatched), default=0.0)
        stalled = abs(worst - previous_worst) <= _STALLED_SHARE * worst
        if not unmatched or stalled or match_pass == _MAXIMUM_MATCH_PASSES - 1:
            break
        previous_worst = worst
        for bone_name in unmatched:
            local_rotations[bone_name] = (
                local_rotations[bone_name] @ np.swapaxes(evaluated_by_bone[bone_name], -1, -2) @ intended[bone_name]
            )
            _write_every_frame(action, armature_object, bone_name, frames, local_rotations[bone_name])
    quaternions_by_bone = {
        bone_name: rotations.make_continuous(rotations.quaternions_from_matrices(local))
        for bone_name, local in local_rotations.items()
    }
    return quaternions_by_bone, tuple(unmatched)


def _intended_rotations(rig: RigRest, local_rotations: dict[str, np.ndarray], frame_count: int) -> dict[str, np.ndarray]:
    identity = np.broadcast_to(np.eye(3), (frame_count, 3, 3))
    globals_by_index: dict[int, np.ndarray] = {}
    for index, bone_name in enumerate(rig.bone_names):
        parent_index = rig.parent_indices[index]
        local = local_rotations.get(bone_name, identity)
        if parent_index < 0:
            globals_by_index[index] = rig.rest_rotations[index] @ local
        else:
            rest_local = rig.rest_rotations[parent_index].T @ rig.rest_rotations[index]
            globals_by_index[index] = globals_by_index[parent_index] @ rest_local @ local
    return {bone_name: globals_by_index[rig.index_of(bone_name)] for bone_name in local_rotations}


@contextmanager
def meshes_left_out(armature_object: bpy.types.Object) -> Iterator[None]:
    hidden = [
        obj
        for obj in bpy.data.objects
        if not obj.hide_viewport
        and (
            obj.parent == armature_object
            or any(modifier.type == "ARMATURE" and modifier.object == armature_object for modifier in getattr(obj, "modifiers", ()))
        )
    ]
    for obj in hidden:
        obj.hide_viewport = True
    try:
        yield
    finally:
        for obj in hidden:
            obj.hide_viewport = False


def _evaluated_rotations(armature_object: bpy.types.Object, bone_names: list[str], frames: np.ndarray) -> dict[str, np.ndarray]:
    scene = bpy.context.scene
    frame_before = scene.frame_current
    evaluated = {bone_name: np.empty((len(frames), 3, 3)) for bone_name in bone_names}
    with meshes_left_out(armature_object):
        try:
            for frame_index, frame in enumerate(frames):
                scene.frame_set(int(frame))
                for bone_name in bone_names:
                    evaluated[bone_name][frame_index] = np.array(armature_object.pose.bones[bone_name].matrix.to_3x3().normalized())
        finally:
            scene.frame_set(frame_before)
    return evaluated


def _largest_angle(first: np.ndarray, second: np.ndarray) -> float:
    cosines = (np.trace(np.swapaxes(first, -1, -2) @ second, axis1=-2, axis2=-1) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosines, -1.0, 1.0)).max())


def _write_every_frame(
    action: bpy.types.Action, armature_object: bpy.types.Object, bone_name: str, frames: np.ndarray, local: np.ndarray
) -> None:
    quaternions = rotations.make_continuous(rotations.quaternions_from_matrices(local))
    data_path, values = _channel_values(armature_object.pose.bones[bone_name], quaternions)
    fcurves = _rotation_fcurves(action, armature_object, bone_name, data_path, values.shape[1])
    set_keys(fcurves, frames, values, np.arange(len(frames)))


def _rotation_fcurves(
    action: bpy.types.Action, armature_object: bpy.types.Object, bone_name: str, data_path: str, channel_count: int
) -> list[bpy.types.FCurve]:
    return [
        action.fcurve_ensure_for_datablock(armature_object, data_path, index=channel_index, group_name=bone_name)
        for channel_index in range(channel_count)
    ]


def write_reduced_keys(
    fcurves: list[bpy.types.FCurve],
    frames: np.ndarray,
    values: np.ndarray,
    key_indices: np.ndarray,
    errors_of: Callable[[np.ndarray], np.ndarray],
    tolerance: float,
) -> None:
    for _refinement_pass in range(_MAXIMUM_REFINEMENT_PASSES):
        set_keys(fcurves, frames, values, key_indices)
        evaluated = np.array([[fcurve.evaluate(frame) for fcurve in fcurves] for frame in frames])
        worst_frames = worst_frames_of_each_run(errors_of(evaluated), tolerance)
        if len(worst_frames) == 0:
            return
        key_indices = np.union1d(key_indices, worst_frames)
    set_keys(fcurves, frames, values, np.arange(len(frames)))


def set_keys(fcurves: list[bpy.types.FCurve], frames: np.ndarray, values: np.ndarray, key_indices: np.ndarray) -> None:
    for channel_index, fcurve in enumerate(fcurves):
        coordinates = np.empty(2 * len(key_indices), dtype=np.float64)
        coordinates[0::2] = frames[key_indices]
        coordinates[1::2] = values[key_indices, channel_index]
        fcurve.keyframe_points.clear()
        fcurve.keyframe_points.add(len(key_indices))
        fcurve.keyframe_points.foreach_set("co", coordinates)
        fcurve.update()


def _as_quaternions(channels: np.ndarray, rotation_mode: str) -> np.ndarray:
    if rotation_mode == "QUATERNION":
        return channels
    if rotation_mode == "AXIS_ANGLE":
        return np.array([tuple(Quaternion(axis_angle[1:], axis_angle[0])) for axis_angle in channels])
    return np.array([tuple(Euler(euler, rotation_mode).to_quaternion()) for euler in channels])


def _channel_values(pose_bone: bpy.types.PoseBone, quaternions: np.ndarray) -> tuple[str, np.ndarray]:
    bone_path = f'pose.bones["{bpy.utils.escape_identifier(pose_bone.name)}"]'
    rotation_mode = pose_bone.rotation_mode
    if rotation_mode == "QUATERNION":
        return f"{bone_path}.rotation_quaternion", quaternions
    if rotation_mode == "AXIS_ANGLE":
        axis_angles = []
        for quaternion in quaternions:
            axis, angle = Quaternion(quaternion).to_axis_angle()
            axis_angles.append((angle, axis.x, axis.y, axis.z))
        return f"{bone_path}.rotation_axis_angle", np.array(axis_angles)
    eulers: list[tuple[float, float, float]] = []
    previous: Euler | None = None
    for quaternion in quaternions:
        euler = Quaternion(quaternion).to_euler(rotation_mode, previous) if previous else Quaternion(quaternion).to_euler(rotation_mode)
        eulers.append(tuple(euler))
        previous = euler
    return f"{bone_path}.rotation_euler", np.array(eulers)
