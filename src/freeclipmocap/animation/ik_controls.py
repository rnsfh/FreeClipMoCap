from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import bpy
import numpy as np

from ..constants.bone_roles import Role
from ..rig.armature_bone_map import read_bone_map
from ..rig.armature_rest import rig_rest_from_armature
from ..solver import rotations
from ..solver.pose_solver import rig_axes
from ..solver.tracking_table import RigAxis
from .action_writer import key_tolerances, meshes_left_out, write_bone_rotations, write_reduced_keys
from .key_reduction import angle_between_rotations, select_key_frames, select_position_key_frames

_IK_CONSTRAINT_NAME = "FreeClipMocap IK"
_ROTATION_CONSTRAINT_NAME = "FreeClipMocap IK Rotation"
_POLE_DISTANCE_SHARE_OF_LIMB = 0.5
_TARGET_SIZE_SHARE_OF_LIMB = 0.12
_POLE_SIZE_SHARE_OF_LIMB = 0.05
_LOCATION_TOLERANCE_SHARE_OF_LIMB = 0.001
_ROTATION_TOLERANCE_DEGREES = 0.25
_UNCHANGED_BY_IK_DEGREES = 0.05


class IkControlsError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class IkLimb:
    name: str
    upper_role: Role
    lower_role: Role
    end_role: Role
    bend_axis: RigAxis

    @property
    def slider_name(self) -> str:
        return f"IK {self.name}"


_LIMBS = (
    IkLimb("Left Leg", Role.LEFT_UP_LEG, Role.LEFT_LEG, Role.LEFT_FOOT, RigAxis.FRONT),
    IkLimb("Right Leg", Role.RIGHT_UP_LEG, Role.RIGHT_LEG, Role.RIGHT_FOOT, RigAxis.FRONT),
    IkLimb("Left Arm", Role.LEFT_ARM, Role.LEFT_FORE_ARM, Role.LEFT_HAND, RigAxis.BACK),
    IkLimb("Right Arm", Role.RIGHT_ARM, Role.RIGHT_FORE_ARM, Role.RIGHT_HAND, RigAxis.BACK),
)


@dataclass(frozen=True, slots=True)
class _MappedLimb:
    limb: IkLimb
    upper: str
    lower: str
    end: str


@dataclass(frozen=True, slots=True)
class _LimbControls:
    mapped: _MappedLimb
    target: bpy.types.Object
    pole: bpy.types.Object


def has_ik_controls(armature_object: bpy.types.Object) -> bool:
    return bool(_existing_controls(armature_object))


def ik_slider_names(armature_object: bpy.types.Object) -> list[str]:
    return [controls.mapped.limb.slider_name for controls in _existing_controls(armature_object)]


def ik_control_actions(armature_object: bpy.types.Object) -> list[bpy.types.Action]:
    actions: list[bpy.types.Action] = []
    for controls in _existing_controls(armature_object):
        for control in (controls.target, controls.pole):
            if control.animation_data is not None and control.animation_data.action is not None:
                actions.append(control.animation_data.action)
    return actions


def add_ik_controls(scene: bpy.types.Scene, armature_object: bpy.types.Object) -> list[str]:
    if armature_object.animation_data is None or armature_object.animation_data.action is None:
        raise IkControlsError("Solve a take first; the IK targets follow its motion.")
    mapped_limbs = _mapped_limbs(armature_object)
    if not mapped_limbs:
        raise IkControlsError("The bone map has no complete arm or leg.")
    already = {controls.mapped.limb.name for controls in _existing_controls(armature_object)}
    rig = rig_rest_from_armature(armature_object)
    axes = rig_axes(rig, read_bone_map(armature_object))
    collection = _controls_collection(scene, armature_object)
    added: list[str] = []
    for mapped in mapped_limbs:
        if mapped.limb.name in already:
            continue
        limb_length = _limb_length(armature_object, mapped)
        target = _new_empty(f"{armature_object.name} IK {mapped.limb.name}", "CUBE", limb_length * _TARGET_SIZE_SHARE_OF_LIMB)
        pole = _new_empty(f"{armature_object.name} Pole {mapped.limb.name}", "SPHERE", limb_length * _POLE_SIZE_SHARE_OF_LIMB)
        for control in (target, pole):
            collection.objects.link(control)
            control.parent = armature_object
        _add_slider(armature_object, mapped.limb.slider_name)
        lower_bone = armature_object.pose.bones[mapped.lower]
        inverse_kinematics = lower_bone.constraints.new("IK")
        inverse_kinematics.name = _IK_CONSTRAINT_NAME
        inverse_kinematics.target = target
        inverse_kinematics.pole_target = pole
        inverse_kinematics.pole_angle = _pole_angle(armature_object.data.bones[mapped.upper], axes[mapped.limb.bend_axis])
        inverse_kinematics.chain_count = 2
        inverse_kinematics.use_tail = True
        inverse_kinematics.use_stretch = False
        copy_rotation = armature_object.pose.bones[mapped.end].constraints.new("COPY_ROTATION")
        copy_rotation.name = _ROTATION_CONSTRAINT_NAME
        copy_rotation.target = target
        for constraint in (inverse_kinematics, copy_rotation):
            _drive_influence(constraint, armature_object, mapped.limb.slider_name)
        added.append(mapped.limb.name)
    key_ik_targets_from_fk(scene, armature_object)
    return added


def key_ik_targets_from_fk(scene: bpy.types.Scene, armature_object: bpy.types.Object) -> None:
    controls_list = _existing_controls(armature_object)
    if not controls_list or armature_object.animation_data is None or armature_object.animation_data.action is None:
        return
    first_frame, last_frame = (int(round(frame)) for frame in armature_object.animation_data.action.frame_range)
    frames = np.arange(first_frame, last_frame + 1, dtype=np.float64)
    rig = rig_rest_from_armature(armature_object)
    axes = rig_axes(rig, read_bone_map(armature_object))
    pose_bones = armature_object.pose.bones
    names = [controls.mapped.limb.name for controls in controls_list]
    roots = {name: np.empty((len(frames), 3)) for name in names}
    middles = {name: np.empty((len(frames), 3)) for name in names}
    tips = {name: np.empty((len(frames), 3)) for name in names}
    upper_rotations = {name: np.empty((len(frames), 3, 3)) for name in names}
    end_rotations = {name: np.empty((len(frames), 3, 3)) for name in names}
    frame_before = scene.frame_current
    with muted_ik_controls(armature_object), meshes_left_out(armature_object):
        try:
            for frame_index, frame in enumerate(frames):
                scene.frame_set(int(frame))
                for controls in controls_list:
                    mapped = controls.mapped
                    name = mapped.limb.name
                    roots[name][frame_index] = tuple(pose_bones[mapped.upper].head)
                    middles[name][frame_index] = tuple(pose_bones[mapped.lower].head)
                    tips[name][frame_index] = tuple(pose_bones[mapped.lower].tail)
                    upper_rotations[name][frame_index] = np.array(pose_bones[mapped.upper].matrix.to_3x3().normalized())
                    end_rotations[name][frame_index] = np.array(pose_bones[mapped.end].matrix.to_3x3().normalized())
        finally:
            scene.frame_set(frame_before)
    for controls in controls_list:
        mapped = controls.mapped
        name = mapped.limb.name
        limb_length = _limb_length(armature_object, mapped)
        bend = upper_rotations[name] @ _bend_direction_in_bone(armature_object.data.bones[mapped.upper], axes[mapped.limb.bend_axis])
        poles = _pole_positions(roots[name], middles[name], tips[name], bend, _POLE_DISTANCE_SHARE_OF_LIMB * limb_length)
        tolerance = _LOCATION_TOLERANCE_SHARE_OF_LIMB * limb_length
        _key_locations(controls.target, frames, tips[name], tolerance)
        _key_rotations(controls.target, frames, rotations.quaternions_from_matrices(end_rotations[name]))
        _key_locations(controls.pole, frames, poles, tolerance)


@contextmanager
def muted_ik_controls(armature_object: bpy.types.Object) -> Iterator[None]:
    constraints = [
        constraint
        for pose_bone in armature_object.pose.bones
        for constraint in pose_bone.constraints
        if constraint.name in (_IK_CONSTRAINT_NAME, _ROTATION_CONSTRAINT_NAME)
    ]
    mutes_before = [constraint.mute for constraint in constraints]
    for constraint in constraints:
        constraint.mute = True
    try:
        yield
    finally:
        for constraint, mute in zip(constraints, mutes_before):
            constraint.mute = mute


def remove_ik_controls(
    scene: bpy.types.Scene, armature_object: bpy.types.Object, keep_pose: bool, key_tolerance_degrees: float | None
) -> list[str]:
    controls_list = _existing_controls(armature_object)
    animation_data = armature_object.animation_data
    action = animation_data.action if animation_data is not None else None
    kept: dict[str, np.ndarray] = {}
    frames = np.zeros(0)
    if keep_pose and action is not None:
        first_frame, last_frame = (int(round(frame)) for frame in action.frame_range)
        frames = np.arange(first_frame, last_frame + 1, dtype=np.float64)
        bone_names = [name for controls in controls_list for name in (controls.mapped.upper, controls.mapped.lower, controls.mapped.end)]
        with_ik = _evaluated_local_rotations(scene, armature_object, bone_names, frames)
        with muted_ik_controls(armature_object):
            without_ik = _evaluated_local_rotations(scene, armature_object, bone_names, frames)
        for controls in controls_list:
            limb_bones = (controls.mapped.upper, controls.mapped.lower, controls.mapped.end)
            if any(_largest_angle_degrees(with_ik[name], without_ik[name]) > _UNCHANGED_BY_IK_DEGREES for name in limb_bones):
                kept.update({name: with_ik[name] for name in limb_bones})
    for controls in controls_list:
        mapped = controls.mapped
        for bone_name, constraint_name in ((mapped.lower, _IK_CONSTRAINT_NAME), (mapped.end, _ROTATION_CONSTRAINT_NAME)):
            pose_bone = armature_object.pose.bones[bone_name]
            constraint = pose_bone.constraints.get(constraint_name)
            if constraint is not None:
                constraint.driver_remove("influence")
                pose_bone.constraints.remove(constraint)
        for control in (controls.target, controls.pole):
            control_action = control.animation_data.action if control.animation_data is not None else None
            bpy.data.objects.remove(control)
            if control_action is not None and control_action.users == 0:
                bpy.data.actions.remove(control_action)
        if mapped.limb.slider_name in armature_object.keys():
            del armature_object[mapped.limb.slider_name]
    collection = bpy.data.collections.get(_collection_name(armature_object))
    if collection is not None and not collection.all_objects:
        bpy.data.collections.remove(collection)
    if kept and action is not None:
        rig = rig_rest_from_armature(armature_object)
        foot_bones = tuple(controls.mapped.end for controls in controls_list if controls.mapped.limb.bend_axis is RigAxis.FRONT)
        tolerances = key_tolerances(rig, list(kept), foot_bones, key_tolerance_degrees) if key_tolerance_degrees is not None else {}
        for bone_name, local in kept.items():
            quaternions = rotations.make_continuous(rotations.quaternions_from_matrices(local))
            write_bone_rotations(armature_object, action, bone_name, frames, quaternions, tolerances.get(bone_name))
    return [controls.mapped.limb.name for controls in controls_list]


def _largest_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosines = (np.trace(np.swapaxes(first, -1, -2) @ second, axis1=-2, axis2=-1) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))).max())


def _mapped_limbs(armature_object: bpy.types.Object) -> list[_MappedLimb]:
    bone_map = read_bone_map(armature_object)
    pose_bones = armature_object.pose.bones
    mapped_limbs: list[_MappedLimb] = []
    for limb in _LIMBS:
        names = [bone_map.bone_for(role) for role in (limb.upper_role, limb.lower_role, limb.end_role)]
        if any(name is None or name not in pose_bones for name in names):
            continue
        upper, lower, end = (str(name) for name in names)
        if pose_bones[lower].parent != pose_bones[upper] or pose_bones[end].parent != pose_bones[lower]:
            continue
        mapped_limbs.append(_MappedLimb(limb, upper, lower, end))
    return mapped_limbs


def _existing_controls(armature_object: bpy.types.Object) -> list[_LimbControls]:
    controls_list: list[_LimbControls] = []
    for mapped in _mapped_limbs(armature_object):
        constraint = armature_object.pose.bones[mapped.lower].constraints.get(_IK_CONSTRAINT_NAME)
        if constraint is None or constraint.target is None or constraint.pole_target is None:
            continue
        controls_list.append(_LimbControls(mapped, constraint.target, constraint.pole_target))
    return controls_list


def _collection_name(armature_object: bpy.types.Object) -> str:
    return f"{armature_object.name} IK Controls"


def _controls_collection(scene: bpy.types.Scene, armature_object: bpy.types.Object) -> bpy.types.Collection:
    name = _collection_name(armature_object)
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if collection not in scene.collection.children_recursive:
        parent = armature_object.users_collection[0] if armature_object.users_collection else scene.collection
        parent.children.link(collection)
    return collection


def _new_empty(name: str, display_type: str, size: float) -> bpy.types.Object:
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = display_type
    empty.empty_display_size = size
    empty.rotation_mode = "QUATERNION"
    empty.show_in_front = True
    return empty


def _add_slider(armature_object: bpy.types.Object, slider_name: str) -> None:
    armature_object[slider_name] = 1.0
    armature_object.id_properties_ui(slider_name).update(
        min=0.0,
        max=1.0,
        soft_min=0.0,
        soft_max=1.0,
        subtype="FACTOR",
        description="0 = FK (the solved bone rotations), 1 = IK (the limb reaches for its target empty). Keyable",
    )


def _drive_influence(constraint: bpy.types.Constraint, armature_object: bpy.types.Object, slider_name: str) -> None:
    driver = constraint.driver_add("influence").driver
    driver.type = "AVERAGE"
    variable = driver.variables.new()
    variable.name = "ik"
    variable.type = "SINGLE_PROP"
    variable.targets[0].id_type = "OBJECT"
    variable.targets[0].id = armature_object
    variable.targets[0].data_path = f'["{slider_name}"]'


def _limb_length(armature_object: bpy.types.Object, mapped: _MappedLimb) -> float:
    bones = armature_object.data.bones
    return float(bones[mapped.upper].length + bones[mapped.lower].length)


def _bend_direction_in_bone(bone: bpy.types.Bone, bend_direction: np.ndarray) -> np.ndarray:
    local = np.array(bone.matrix_local.to_3x3().normalized()).T @ bend_direction
    local[1] = 0.0
    return local / np.linalg.norm(local)


def _pole_angle(upper_bone: bpy.types.Bone, bend_direction: np.ndarray) -> float:
    local = _bend_direction_in_bone(upper_bone, bend_direction)
    return float(np.arctan2(local[2], local[0]))


def _pole_positions(
    roots: np.ndarray, middles: np.ndarray, tips: np.ndarray, bends: np.ndarray, distance: float
) -> np.ndarray:
    # Blender's IK turns the chain until the root bone's axis at the pole angle (here the bend axis) points, across
    # the root-to-target line, the same way as the pole seen from the target. A pole straight out from that line
    # along the bend axis therefore keeps the solved FK pose exactly; it is placed level with the knee or elbow.
    along = rotations.normalized(tips - roots)
    across = rotations.normalized(bends - np.sum(bends * along, axis=-1, keepdims=True) * along)
    level_with_middle = roots + np.sum((middles - roots) * along, axis=-1, keepdims=True) * along
    return level_with_middle + distance * across


def _control_fcurves(control: bpy.types.Object, data_path: str, channel_count: int) -> list[bpy.types.FCurve]:
    animation_data = control.animation_data or control.animation_data_create()
    if animation_data.action is None:
        animation_data.action = bpy.data.actions.new(control.name)
    return [
        animation_data.action.fcurve_ensure_for_datablock(control, data_path, index=channel_index)
        for channel_index in range(channel_count)
    ]


def _key_locations(control: bpy.types.Object, frames: np.ndarray, locations: np.ndarray, tolerance: float) -> None:
    write_reduced_keys(
        _control_fcurves(control, "location", 3),
        frames,
        locations,
        select_position_key_frames(locations, tolerance),
        lambda evaluated: np.linalg.norm(evaluated - locations, axis=-1),
        tolerance,
    )


def _key_rotations(control: bpy.types.Object, frames: np.ndarray, quaternions: np.ndarray) -> None:
    continuous = rotations.make_continuous(quaternions)
    fcurves = _control_fcurves(control, "rotation_quaternion", 4)
    write_reduced_keys(
        fcurves,
        frames,
        continuous,
        select_key_frames(continuous, _ROTATION_TOLERANCE_DEGREES),
        lambda evaluated: angle_between_rotations(evaluated, continuous),
        np.radians(_ROTATION_TOLERANCE_DEGREES),
    )


def _evaluated_local_rotations(
    scene: bpy.types.Scene, armature_object: bpy.types.Object, bone_names: list[str], frames: np.ndarray
) -> dict[str, np.ndarray]:
    pose_bones = armature_object.pose.bones
    sampled_names = set(bone_names) | {pose_bones[name].parent.name for name in bone_names if pose_bones[name].parent}
    evaluated = {name: np.empty((len(frames), 3, 3)) for name in sampled_names}
    frame_before = scene.frame_current
    with meshes_left_out(armature_object):
        try:
            for frame_index, frame in enumerate(frames):
                scene.frame_set(int(frame))
                for name in sampled_names:
                    evaluated[name][frame_index] = np.array(pose_bones[name].matrix.to_3x3().normalized())
        finally:
            scene.frame_set(frame_before)
    locals_by_bone: dict[str, np.ndarray] = {}
    for name in bone_names:
        bone = armature_object.data.bones[name]
        rest = np.array(bone.matrix_local.to_3x3().normalized())
        if bone.parent is None:
            locals_by_bone[name] = rest.T @ evaluated[name]
            continue
        rest_local = np.array(bone.parent.matrix_local.to_3x3().normalized()).T @ rest
        locals_by_bone[name] = rest_local.T @ np.swapaxes(evaluated[bone.parent.name], -1, -2) @ evaluated[name]
    return locals_by_bone
