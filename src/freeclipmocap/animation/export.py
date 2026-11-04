from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import bpy
import numpy as np

from ..rig.armature_rest import rig_rest_from_armature
from .ik_controls import ik_control_actions
from .bvh_writer import write_bvh


@contextmanager
def _only_selected(context: bpy.types.Context, armature_object: bpy.types.Object) -> Iterator[None]:
    view_layer = context.view_layer
    previously_selected = [obj for obj in view_layer.objects if obj.select_get()]
    previously_active = view_layer.objects.active
    for obj in previously_selected:
        obj.select_set(False)
    armature_object.select_set(True)
    view_layer.objects.active = armature_object
    try:
        yield
    finally:
        armature_object.select_set(False)
        for obj in previously_selected:
            obj.select_set(True)
        view_layer.objects.active = previously_active


def _shift_keys(action: bpy.types.Action, frames: float) -> None:
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                for fcurve in channelbag.fcurves:
                    points = fcurve.keyframe_points
                    for attribute in ("co", "handle_left", "handle_right"):
                        values = np.empty(2 * len(points), dtype=np.float64)
                        points.foreach_get(attribute, values)
                        values[0::2] += frames
                        points.foreach_set(attribute, values)
                    fcurve.update()


@contextmanager
def take_named_and_starting_at_zero(scene: bpy.types.Scene, armature_object: bpy.types.Object) -> Iterator[None]:
    action = armature_object.animation_data.action
    offset = scene.frame_start
    scene_name = scene.name
    shifted = [action, *ik_control_actions(armature_object)]
    for shifted_action in shifted:
        _shift_keys(shifted_action, -offset)
    scene.frame_start -= offset
    scene.frame_end -= offset
    scene.name = action.name
    try:
        yield
    finally:
        scene.name = scene_name
        scene.frame_end += offset
        scene.frame_start += offset
        for shifted_action in shifted:
            _shift_keys(shifted_action, offset)


def export_fbx(context: bpy.types.Context, armature_object: bpy.types.Object, path: Path) -> None:
    with (
        _only_selected(context, armature_object),
        take_named_and_starting_at_zero(context.scene, armature_object),
    ):
        bpy.ops.export_scene.fbx(
            filepath=str(path),
            use_selection=True,
            object_types={"ARMATURE"},
            global_scale=0.01,
            apply_unit_scale=True,
            apply_scale_options="FBX_SCALE_ALL",
            axis_forward="-Z",
            axis_up="Y",
            primary_bone_axis="Y",
            secondary_bone_axis="X",
            armature_nodetype="NULL",
            use_armature_deform_only=False,
            add_leaf_bones=False,
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_use_nla_strips=False,
            bake_anim_use_all_actions=False,
            bake_anim_force_startend_keying=True,
            bake_anim_simplify_factor=0.0,
        )


def export_bvh(context: bpy.types.Context, armature_object: bpy.types.Object, path: Path) -> None:
    scene = context.scene
    rig = rig_rest_from_armature(armature_object)
    pose_bones = [armature_object.pose.bones[bone_name] for bone_name in rig.bone_names]
    frames = range(scene.frame_start, scene.frame_end + 1)
    pose_rotations = np.empty((len(frames), len(pose_bones), 3, 3), dtype=np.float64)
    pose_heads = np.empty((len(frames), len(pose_bones), 3), dtype=np.float64)
    frame_before = scene.frame_current
    try:
        for frame_index, frame in enumerate(frames):
            scene.frame_set(frame)
            for bone_index, pose_bone in enumerate(pose_bones):
                pose_rotations[frame_index, bone_index] = np.array(pose_bone.matrix.to_3x3().normalized())
                pose_heads[frame_index, bone_index] = tuple(pose_bone.head)
    finally:
        scene.frame_set(frame_before)
    write_bvh(path, rig, pose_rotations, pose_heads, scene.render.fps_base / scene.render.fps)
