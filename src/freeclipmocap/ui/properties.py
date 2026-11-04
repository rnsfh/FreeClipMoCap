import bpy

from ..capture.capture_settings import CaptureSettings, PoseModelVariant
from ..constants import calibration_board
from ..rig.armature_bone_map import available_presets, guess_and_write_bone_map, has_bone_map
from .lens_folder import available_lenses
from .video_overlay import redraw_clip_editors

_preset_items: list[tuple[str, str, str]] = []
MEASURE_LENS_IDENTIFIER = "_MEASURE"
TYPED_LENS_IDENTIFIER = "_TYPED"
_MEASURE_LENS_CHOICE = (MEASURE_LENS_IDENTIFIER, "Measure", "Measure this phone's lens from the board held close to it", 0)
_TYPED_LENS_CHOICE = (TYPED_LENS_IDENTIFIER, "Typed", "Enter the focal length in mm (35 mm equivalent, as phone specs give it)", 1)
_kept_lens_items: dict[tuple[tuple[str, str, str, int], ...], list[tuple[str, str, str, int]]] = {}


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


def lens_choices(lens_name: str) -> list[tuple[str, str, str, int]]:
    lenses = available_lenses()
    choices = [_MEASURE_LENS_CHOICE, _TYPED_LENS_CHOICE]
    for lens in lenses:
        description = f"Saved lens for {lens.image_width}×{lens.image_height} video, kept fixed while calibrating"
        choices.append((lens.name, lens.name, description, len(choices)))
    if lens_name and all(lens.name != lens_name for lens in lenses):
        description = "This saved lens is no longer in your FreeClipMocap user folder"
        choices.append((lens_name, f"{lens_name} (missing)", description, len(choices)))
    return choices


def calibration_lens_items(self: "CalibrationLensSettings", context: bpy.types.Context) -> list[tuple[str, str, str, int]]:
    choices = lens_choices(self.calibration_lens_name)
    # Blender keeps pointers into the item strings it was handed, so every list handed out stays referenced.
    return _kept_lens_items.setdefault(tuple(choices), choices)


def get_calibration_lens(self: "CalibrationLensSettings") -> int:
    if self.calibration_lens_is_typed:
        chosen_identifier = TYPED_LENS_IDENTIFIER
    else:
        chosen_identifier = self.calibration_lens_name or MEASURE_LENS_IDENTIFIER
    return next(
        (number for identifier, _label, _description, number in lens_choices(self.calibration_lens_name) if identifier == chosen_identifier),
        0,
    )


def set_calibration_lens(self: "CalibrationLensSettings", chosen_number: int) -> None:
    chosen_identifier = next(
        (identifier for identifier, _label, _description, number in lens_choices(self.calibration_lens_name) if number == chosen_number),
        MEASURE_LENS_IDENTIFIER,
    )
    self.calibration_lens_is_typed = chosen_identifier == TYPED_LENS_IDENTIFIER
    self.calibration_lens_name = "" if chosen_identifier in (MEASURE_LENS_IDENTIFIER, TYPED_LENS_IDENTIFIER) else chosen_identifier


class CalibrationLensSettings:
    calibration_lens_name: bpy.props.StringProperty(
        name="Lens Name",
        description="Name of the saved lens this camera keeps while calibrating; empty: measured or typed",
    )
    calibration_lens_is_typed: bpy.props.BoolProperty(
        name="Typed Lens",
        description="This camera's focal length is typed in instead of measured",
        default=False,
    )
    calibration_focal_length_millimetres: bpy.props.FloatProperty(
        name="Focal Length",
        description="This phone's focal length in mm, 35 mm equivalent as phone specs give it (across the video's "
        "longer side, like the Compare View lens); 26 mm is a typical main camera",
        min=8.0,
        max=600.0,
        default=26.0,
        precision=1,
        subtype="DISTANCE_CAMERA",
        unit="CAMERA",
    )
    calibration_keep_focal_length: bpy.props.BoolProperty(
        name="Keep Fixed",
        description="Use exactly this focal length with no lens distortion, so no close-ups are needed. Off: it is "
        "where the calibration starts, and the board corrects it (a spec sheet's value is rounded and meant for photos)",
        default=False,
    )
    calibration_lens: bpy.props.EnumProperty(
        name="Lens",
        description="Measure this phone's lens from the board held close to it, type its focal length, or keep a lens "
        "saved from an earlier calibration (same phone, lens, resolution, orientation and zoom)",
        items=calibration_lens_items,
        get=get_calibration_lens,
        set=set_calibration_lens,
    )


class BoneMapEntry(bpy.types.PropertyGroup):
    role: bpy.props.StringProperty(name="Role")
    bone_name: bpy.props.StringProperty(name="Bone", description="Bone of the armature that plays this role")


class RigProperties(bpy.types.PropertyGroup):
    bone_map: bpy.props.CollectionProperty(type=BoneMapEntry)
    rig_settings_json: bpy.props.StringProperty(
        name="Rig Settings",
        description="Settings of the rig's own controls (such as IK/FK switches) that Solve applies, from the bone map preset",
    )


class CameraVideo(bpy.types.PropertyGroup, CalibrationLensSettings):
    video_path: bpy.props.StringProperty(
        name="Video",
        description="This camera's video of the take",
        subtype="FILE_PATH",
    )
    started_later_seconds: bpy.props.FloatProperty(
        name="Sync",
        description="How many seconds after the first camera this camera started recording (negative: before). "
        "Sync by Sound fills it in",
        precision=3,
        step=1,
        subtype="TIME_ABSOLUTE",
        unit="TIME_ABSOLUTE",
    )
    calibration_video_path: bpy.props.StringProperty(
        name="Calibration Video",
        description="This camera's video of the calibration board",
        subtype="FILE_PATH",
    )


class FreeClipMocapProperties(bpy.types.PropertyGroup, CalibrationLensSettings):
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
    extra_videos: bpy.props.CollectionProperty(type=CameraVideo)
    calibration_video_path: bpy.props.StringProperty(
        name="Calibration Video",
        description="The first camera's video of the calibration board",
        subtype="FILE_PATH",
    )
    use_separate_calibration_videos: bpy.props.BoolProperty(
        name="Separate Calibration Videos",
        description="Calibrate from other videos (filmed once, reused while the cameras stay put) instead of the take's "
        "own videos",
        default=False,
    )
    calibration_method: bpy.props.EnumProperty(
        name="Method",
        description="What the cameras are calibrated from",
        items=(
            ("BOARD", "Board", "The printed or shown board: the most accurate"),
            (
                "BODY",
                "Body",
                "The actor walking around: no board needed, less accurate. Needs each camera's focal length (typed or "
                "saved) and the measured distance between the phones",
            ),
        ),
        default="BOARD",
    )
    cameras_apart_metres: bpy.props.FloatProperty(
        name="Phones Apart",
        description="Distance between camera 1's and camera 2's lenses, measured with a tape; it sets the scale",
        min=0.1,
        max=50.0,
        default=2.5,
        precision=3,
        subtype="DISTANCE",
        unit="LENGTH",
    )
    calibration_start_frame: bpy.props.IntProperty(
        name="Start",
        description="First frame of camera 1's video to calibrate from: the board, or the actor walking around",
        min=1,
        default=1,
    )
    calibration_end_frame: bpy.props.IntProperty(
        name="End",
        description="Last frame of camera 1's video to calibrate from (the board, or the actor walking around); 0 means "
        "the last frame",
        min=0,
        default=0,
    )
    board_square_millimetres: bpy.props.FloatProperty(
        name="Square",
        description="Size of one square of the board, measured with a ruler (printers and screens scale it)",
        min=5.0,
        max=500.0,
        default=calibration_board.NOMINAL_SQUARE_MILLIMETRES,
        precision=1,
    )
    paper_size: bpy.props.EnumProperty(
        name="Paper",
        description="What the board is printed on or shown on; the floor is found from its bottom edge",
        items=(
            ("A4", "A4", "297 mm high, the board centred"),
            ("LETTER", "Letter", "279.4 mm high, the board centred"),
            ("CUSTOM", "Custom", "A tablet or laptop screen, or anything else: enter the distance below the squares"),
        ),
        default="A4",
    )
    board_edge_below_squares_millimetres: bpy.props.FloatProperty(
        name="Below Squares",
        description="Distance from the lowest row of squares down to the edge that stands on the floor (a screen's "
        "bezel included), measured with a ruler",
        min=0.0,
        max=500.0,
        default=20.0,
        precision=1,
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
    use_all_cameras: bpy.props.BoolProperty(
        name="All Cameras",
        description="Solve from every camera's video, its points triangulated through the calibration (needs Capture "
        "of every video and Calibrate with the current Sync). Off: camera 1 only, depth guessed by the tracker",
        default=True,
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
    compare_lens_millimetres: bpy.props.FloatProperty(
        name="Lens",
        description="Focal length of the camera that filmed the video, 35 mm equivalent; 26 mm is a phone's main camera. "
        "Used by Compare View, and by Solve from one camera to straighten the legs by the video",
        min=8.0,
        max=600.0,
        default=26.0,
        subtype="DISTANCE_CAMERA",
        unit="CAMERA",
    )
    compare_camera: bpy.props.PointerProperty(
        name="Compare Camera",
        description="Camera made by Compare View; pressing it again updates this camera",
        type=bpy.types.Object,
    )
    bone_map_preset: bpy.props.EnumProperty(name="Preset", description="Bone map preset", items=bone_map_preset_items)
    show_bone_map: bpy.props.BoolProperty(name="Bone Map", default=False)
    show_left_fingers: bpy.props.BoolProperty(name="Left Fingers", default=False)
    show_right_fingers: bpy.props.BoolProperty(name="Right Fingers", default=False)

    def capture_settings(self) -> CaptureSettings:
        return CaptureSettings(pose_model_variant=PoseModelVariant(self.pose_model_variant))

    def edge_below_squares_metres(self) -> float:
        if self.paper_size == "CUSTOM":
            return self.board_edge_below_squares_millimetres / 1000.0
        if self.paper_size == "LETTER":
            paper_height_millimetres = calibration_board.LETTER_PAPER_HEIGHT_MILLIMETRES
        else:
            paper_height_millimetres = calibration_board.A4_PAPER_HEIGHT_MILLIMETRES
        board_height_millimetres = calibration_board.SQUARES_DOWN * self.board_square_millimetres
        return (paper_height_millimetres - board_height_millimetres) / 2.0 / 1000.0


_classes = (BoneMapEntry, RigProperties, CameraVideo, FreeClipMocapProperties)


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
