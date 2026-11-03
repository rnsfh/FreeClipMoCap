from collections.abc import Sequence

import bpy

from ..constants.bone_roles import BODY_ROLES, LEFT_FINGER_ROLES, RIGHT_FINGER_ROLES, Role
from ..rig.armature_bone_map import read_bone_map
from .bone_map_operators import (
    ApplyBoneMapPresetOperator,
    DeleteBoneMapPresetOperator,
    GuessBoneMapOperator,
    SaveBoneMapPresetOperator,
    describe_bone_map,
)
from .calibration_operators import (
    AddCameraVideoOperator,
    CalibrateOperator,
    CancelCalibrationOperator,
    DeleteLensOperator,
    RemoveCameraVideoOperator,
    SaveBoardImageOperator,
    SaveLensOperator,
    SyncBySoundOperator,
    calibrated_camera,
    calibration_state,
    camera_settings,
    lens_is_missing_for_body,
    uses_body,
)
from .capture_operators import CancelCaptureOperator, CaptureOperator, capture_state
from .compare_view import CompareViewOperator
from .export_operators import ExportBvhOperator, ExportFbxOperator, current_take
from .solve_operator import SolveOperator
from .video_operators import SetVideoFrameOperator, ShowVideoOperator


class MainPanel(bpy.types.Panel):
    bl_idname = "FREECLIPMOCAP_PT_main"
    bl_label = "FreeClipMocap"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FreeClipMocap"

    def draw(self, context: bpy.types.Context) -> None:
        properties = context.scene.freeclipmocap
        layout = self.layout

        video_box = layout.box()
        video_box.label(text="Video", icon="FILE_MOVIE")
        self._draw_camera_videos(video_box, properties)
        video_row = video_box.row()
        video_row.operator(ShowVideoOperator.bl_idname, icon="SEQ_PREVIEW")
        video_row.prop(properties, "show_tracked_skeleton")
        frame_row = video_box.row(align=True)
        frame_row.prop(properties, "video_start_frame")
        frame_row.operator(SetVideoFrameOperator.bl_idname, text="", icon="EYEDROPPER").edge = "START"
        frame_row.separator()
        frame_row.prop(properties, "video_end_frame")
        frame_row.operator(SetVideoFrameOperator.bl_idname, text="", icon="EYEDROPPER").edge = "END"

        if properties.extra_videos:
            self._draw_calibration(layout.box(), context, properties)
        self._draw_target(layout.box(), properties)
        self._draw_capture(layout.box(), properties)

        solve_box = layout.box()
        solve_box.label(text="Solve", icon="POSE_HLT")
        solve_box.prop(properties, "smoothing", slider=True)
        solve_box.prop(properties, "root_motion")
        key_row = solve_box.row()
        key_row.prop(properties, "reduce_keys")
        tolerance_column = key_row.column()
        tolerance_column.enabled = properties.reduce_keys
        tolerance_column.prop(properties, "key_tolerance_degrees")
        solve_box.operator(SolveOperator.bl_idname, icon="ACTION")
        compare_row = solve_box.row(align=True)
        compare_row.operator(CompareViewOperator.bl_idname, icon="VIEW_CAMERA")
        compare_row.prop(properties, "compare_lens_millimetres")

        export_box = layout.box()
        export_box.label(text="Export", icon="EXPORT")
        take = current_take(context)
        export_box.label(text=f"Take: {take.name}" if take else "Solve first; the current take is exported.")
        export_row = export_box.row(align=True)
        export_row.operator(ExportFbxOperator.bl_idname, text="FBX")
        export_row.operator(ExportBvhOperator.bl_idname, text="BVH")

    def _draw_camera_videos(self, box: bpy.types.UILayout, properties: bpy.types.PropertyGroup) -> None:
        if not properties.extra_videos:
            box.prop(properties, "video_path", text="")
        else:
            first_row = box.row(align=True)
            first_row.label(text="1")
            first_row.prop(properties, "video_path", text="")
            for index, camera in enumerate(properties.extra_videos):
                row = box.row(align=True)
                row.label(text=str(index + 2))
                row.prop(camera, "video_path", text="")
                row.prop(camera, "started_later_seconds", text="")
                row.operator(RemoveCameraVideoOperator.bl_idname, text="", icon="X").index = index
        list_row = box.row(align=True)
        list_row.operator(AddCameraVideoOperator.bl_idname, icon="ADD")
        if properties.extra_videos:
            list_row.operator(SyncBySoundOperator.bl_idname, icon="SOUND")

    def _draw_calibration(self, box: bpy.types.UILayout, context: bpy.types.Context, properties: bpy.types.PropertyGroup) -> None:
        box.label(text="Calibration", icon="CON_CAMERASOLVER")
        settings_column = box.column()
        settings_column.enabled = not calibration_state.is_running
        settings_column.row().prop(properties, "calibration_method", expand=True)
        if uses_body(properties):
            settings_column.prop(properties, "cameras_apart_metres")
        else:
            settings_column.operator(SaveBoardImageOperator.bl_idname, icon="FILE_IMAGE")
            board_row = settings_column.row(align=True)
            board_row.prop(properties, "board_square_millimetres")
            board_row.prop(properties, "paper_size", text="")
            if properties.paper_size == "CUSTOM":
                settings_column.prop(properties, "board_edge_below_squares_millimetres")
        self._draw_lenses(settings_column, context, properties)
        settings_column.prop(properties, "use_separate_calibration_videos")
        if properties.use_separate_calibration_videos:
            first_row = settings_column.row(align=True)
            first_row.label(text="1")
            first_row.prop(properties, "calibration_video_path", text="")
            for index, camera in enumerate(properties.extra_videos):
                row = settings_column.row(align=True)
                row.label(text=str(index + 2))
                row.prop(camera, "calibration_video_path", text="")
        frame_row = settings_column.row(align=True)
        frame_row.prop(properties, "calibration_start_frame")
        frame_row.prop(properties, "calibration_end_frame")
        job = calibration_state.job
        if job is not None and calibration_state.is_running:
            progress = job.progress
            current = progress.current
            text = f"Camera {progress.camera_index + 1} of {progress.camera_count}: frame {current.frames_done}"
            if current.expected_frame_count > 0:
                text += f" of {current.expected_frame_count}"
            box.progress(factor=progress.fraction, type="BAR", text=text)
            box.operator(CancelCalibrationOperator.bl_idname, icon="CANCEL")
        else:
            box.operator(CalibrateOperator.bl_idname, icon="PLAY")
        if calibration_state.status_message:
            message_column = box.column(align=True)
            for line in _wrapped(calibration_state.status_message):
                message_column.label(text=line)

    def _draw_lenses(self, column: bpy.types.UILayout, context: bpy.types.Context, properties: bpy.types.PropertyGroup) -> None:
        column.label(text="Lenses")
        for camera_index in range(len(properties.extra_videos) + 1):
            settings = camera_settings(properties, camera_index)
            row = column.row(align=True)
            row.label(text=str(camera_index + 1))
            row.prop(settings, "calibration_lens", text="")
            camera = calibrated_camera(context, camera_index)
            save_column = row.column(align=True)
            save_column.enabled = camera is not None
            save_column.operator(SaveLensOperator.bl_idname, text="", icon="FILE_TICK").camera_index = camera_index
            delete_column = row.column(align=True)
            delete_column.enabled = bool(settings.calibration_lens_name)
            delete_column.operator(DeleteLensOperator.bl_idname, text="", icon="TRASH").camera_index = camera_index
            if settings.calibration_lens_is_typed:
                typed_row = column.row(align=True)
                typed_row.separator(factor=2.0)
                typed_row.prop(settings, "calibration_focal_length_millimetres", text="")
                if not uses_body(properties):
                    typed_row.prop(settings, "calibration_keep_focal_length", toggle=True)
            elif uses_body(properties) and lens_is_missing_for_body(settings):
                missing_row = column.row()
                missing_row.alert = True
                missing_row.separator(factor=2.0)
                missing_row.label(text="Type the focal length or pick a saved lens", icon="ERROR")
            if camera is not None:
                measured_row = column.row()
                measured_row.separator(factor=2.0)
                measured_row.label(text=f"Last calibration: {camera.focal_length_millimetres:.1f} mm")

    def _draw_target(self, box: bpy.types.UILayout, properties: bpy.types.PropertyGroup) -> None:
        box.label(text="Target", icon="ARMATURE_DATA")
        box.prop(properties, "target_armature", text="")
        armature_object = properties.target_armature
        if armature_object is None:
            return
        preset_row = box.row(align=True)
        preset_row.prop(properties, "bone_map_preset", text="")
        preset_row.operator(ApplyBoneMapPresetOperator.bl_idname, text="", icon="IMPORT")
        preset_row.operator(SaveBoneMapPresetOperator.bl_idname, text="", icon="ADD")
        preset_row.operator(DeleteBoneMapPresetOperator.bl_idname, text="", icon="TRASH")
        box.operator(GuessBoneMapOperator.bl_idname, icon="VIEWZOOM")

        bone_map = read_bone_map(armature_object)
        status_row = box.row()
        status_row.alert = bool(bone_map.missing_required_roles())
        status_row.label(text=describe_bone_map(bone_map))

        box.prop(properties, "show_bone_map", icon="TRIA_DOWN" if properties.show_bone_map else "TRIA_RIGHT", emboss=False)
        if not properties.show_bone_map:
            return
        entries_by_role = {entry.role: entry for entry in armature_object.freeclipmocap_rig.bone_map}
        self._draw_roles(box.column(align=True), armature_object, entries_by_role, BODY_ROLES)
        for flag_name, roles in (("show_left_fingers", LEFT_FINGER_ROLES), ("show_right_fingers", RIGHT_FINGER_ROLES)):
            is_shown = getattr(properties, flag_name)
            box.prop(properties, flag_name, icon="TRIA_DOWN" if is_shown else "TRIA_RIGHT", emboss=False)
            if is_shown:
                self._draw_roles(box.column(align=True), armature_object, entries_by_role, roles)

    def _draw_roles(
        self,
        column: bpy.types.UILayout,
        armature_object: bpy.types.Object,
        entries_by_role: dict[str, bpy.types.PropertyGroup],
        roles: Sequence[Role],
    ) -> None:
        for role in roles:
            entry = entries_by_role.get(role.value)
            if entry is None:
                column.label(text=f"{role.value}: press Guess Bone Map")
                continue
            column.prop_search(entry, "bone_name", armature_object.data, "bones", text=role.value)

    def _draw_capture(self, box: bpy.types.UILayout, properties: bpy.types.PropertyGroup) -> None:
        box.label(text="Capture", icon="OUTLINER_OB_CAMERA")
        settings_column = box.column()
        settings_column.enabled = not capture_state.is_running
        settings_column.prop(properties, "pose_model_variant")
        job = capture_state.job
        if job is not None and capture_state.is_running:
            progress = job.progress
            if progress.expected_frame_count > 0:
                text = f"Frame {progress.frames_done} of {progress.expected_frame_count}"
            else:
                text = f"Frame {progress.frames_done}"
            box.progress(factor=progress.fraction, type="BAR", text=text)
            box.operator(CancelCaptureOperator.bl_idname, icon="CANCEL")
        else:
            box.operator(CaptureOperator.bl_idname, icon="PLAY")
        if capture_state.status_message:
            box.label(text=capture_state.status_message)


def _wrapped(message: str) -> list[str]:
    separated = message.replace("; ", ";\n").replace(". ", ".\n")
    return [line.strip() for line in separated.splitlines() if line.strip()]


register, unregister = bpy.utils.register_classes_factory((MainPanel,))
