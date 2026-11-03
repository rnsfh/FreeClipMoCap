import bpy

from ..capture.capture_settings import CaptureSettings, PoseModelVariant
from ..rig.armature_bone_map import available_presets, guess_and_write_bone_map, has_bone_map
from .video_overlay import redraw_clip_editors

_preset_items: list[tuple[str, str, str]] = []


def is_armature(self: bpy.types.PropertyGroup, candidate: bpy.types.Object) -> bool:
    return candidate.type == "ARMATURE"


def on_target_armature_changed(self: "FreeClipMocapProperties", context: bpy.types.Context) -> None:
    armature_object = self.target_armature
    if armature_object is not None and not has_bone_map(armature_object):
        guess_and_write_bone_map(armature_object)


def on_video_path_changed(self: "FreeClipMocapProperties", context: bpy.types.Context) -> None:
    self.video_start_frame = 1
    self.video_end_frame = 0


def bone_map_preset_items(self: "FreeClipMocapProperties", context: bpy.types.Context) -> list[tuple[str, str, str]]:
    presets, _problems = available_presets()
    _preset_items.clear()
    for preset in presets:
        description = "Built-in preset" if preset.is_built_in else "Your preset"
        _preset_items.append((preset.name, preset.name, description))
    if not _preset_items:
        _preset_items.append(("", "No presets", ""))
    return _preset_items


class BoneMapEntry(bpy.types.PropertyGroup):
    role: bpy.props.StringProperty(name="Role")
    bone_name: bpy.props.StringProperty(name="Bone", description="Bone of the armature that plays this role")


class RigProperties(bpy.types.PropertyGroup):
    bone_map: bpy.props.CollectionProperty(type=BoneMapEntry)
    rig_settings_json: bpy.props.StringProperty(
        name="Rig Settings",
        description="Settings of the rig's own controls (such as IK/FK switches) that Solve applies, from the bone map preset",
    )


class FreeClipMocapProperties(bpy.types.PropertyGroup):
    video_path: bpy.props.StringProperty(
        name="Video",
        description="Video to capture the motion from",
        subtype="FILE_PATH",
        update=on_video_path_changed,
    )
    video_start_frame: bpy.props.IntProperty(
        name="Start",
        description="First video frame used by Solve. Video frame N is shown at scene frame N",
        min=1,
        default=1,
    )
    video_end_frame: bpy.props.IntProperty(
        name="End",
        description="Last video frame used by Solve; 0 means the last frame of the video",
        min=0,
        default=0,
    )
    pose_model_variant: bpy.props.EnumProperty(
        name="Pose Model",
        description="MediaPipe pose model; lite and full are faster previews",
        items=(
            (PoseModelVariant.HEAVY.value, "Heavy", "Most accurate, slowest"),
            (PoseModelVariant.FULL.value, "Full", "Faster preview"),
            (PoseModelVariant.LITE.value, "Lite", "Fastest preview"),
        ),
        default=PoseModelVariant.HEAVY.value,
    )
    target_armature: bpy.props.PointerProperty(
        name="Armature",
        description="Armature to animate; its bone map is guessed the first time it is picked",
        type=bpy.types.Object,
        poll=is_armature,
        update=on_target_armature_changed,
    )
    smoothing: bpy.props.FloatProperty(
        name="Smoothing",
        description="How strongly jitter is smoothed out; fast moves stay sharp at any setting",
        min=0.0,
        max=1.0,
        default=0.5,
        subtype="FACTOR",
    )
    show_tracked_skeleton: bpy.props.BoolProperty(
        name="Tracked Skeleton",
        description="Draw what the tracker found over the video: body green, left hand orange, right hand blue, "
        "points it marked hidden grey",
        default=True,
        update=lambda self, context: redraw_clip_editors(context),
    )
    root_motion: bpy.props.BoolProperty(
        name="Root Motion",
        description="Let the body travel across the floor as in the video. Off: in place, the hips still move up "
        "and down and sway, but the travel is taken out",
        default=False,
    )
    reduce_keys: bpy.props.BoolProperty(
        name="Fewer Keys",
        description="Keep only the keys needed to stay within the tolerance, so the take is easier to edit by hand",
        default=True,
    )
    key_tolerance_degrees: bpy.props.FloatProperty(
        name="Tolerance",
        description="Largest rotation difference (degrees) allowed between the reduced keys and the full solve",
        min=0.05,
        max=5.0,
        default=2.0,
        precision=2,
    )
    bone_map_preset: bpy.props.EnumProperty(name="Preset", description="Bone map preset", items=bone_map_preset_items)
    show_bone_map: bpy.props.BoolProperty(name="Bone Map", default=False)
    show_left_fingers: bpy.props.BoolProperty(name="Left Fingers", default=False)
    show_right_fingers: bpy.props.BoolProperty(name="Right Fingers", default=False)

    def capture_settings(self) -> CaptureSettings:
        return CaptureSettings(pose_model_variant=PoseModelVariant(self.pose_model_variant))


_classes = (BoneMapEntry, RigProperties, FreeClipMocapProperties)


def register() -> None:
    for property_class in _classes:
        bpy.utils.register_class(property_class)
    bpy.types.Object.freeclipmocap_rig = bpy.props.PointerProperty(type=RigProperties)
    bpy.types.Scene.freeclipmocap = bpy.props.PointerProperty(type=FreeClipMocapProperties)


def unregister() -> None:
    del bpy.types.Scene.freeclipmocap
    del bpy.types.Object.freeclipmocap_rig
    for property_class in reversed(_classes):
        bpy.utils.unregister_class(property_class)
