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
        video_box.prop(properties, "video_path", text="")
        video_row = video_box.row()
        video_row.operator(ShowVideoOperator.bl_idname, icon="SEQ_PREVIEW")
        video_row.prop(properties, "show_tracked_skeleton")
        frame_row = video_box.row(align=True)
        frame_row.prop(properties, "video_start_frame")
        frame_row.operator(SetVideoFrameOperator.bl_idname, text="", icon="EYEDROPPER").edge = "START"
        frame_row.separator()
        frame_row.prop(properties, "video_end_frame")
        frame_row.operator(SetVideoFrameOperator.bl_idname, text="", icon="EYEDROPPER").edge = "END"

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


register, unregister = bpy.utils.register_classes_factory((MainPanel,))
