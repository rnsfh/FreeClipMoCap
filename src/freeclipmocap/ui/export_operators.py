from pathlib import Path

import bpy
from bpy_extras.io_utils import ExportHelper

from ..animation.export import export_bvh, export_fbx


def current_take(context: bpy.types.Context) -> bpy.types.Action | None:
    armature_object = context.scene.freeclipmocap.target_armature
    if armature_object is None or armature_object.animation_data is None:
        return None
    return armature_object.animation_data.action


def default_export_path(context: bpy.types.Context, extension: str) -> str:
    take = current_take(context)
    file_name = f"{bpy.path.clean_name(take.name) if take else 'take'}{extension}"
    video_path = context.scene.freeclipmocap.video_path
    folder = Path(bpy.path.abspath(video_path)).parent if video_path else Path(bpy.path.abspath("//"))
    return str(folder / file_name)


class _ExportTakeOperator(ExportHelper):
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return current_take(context) is not None and context.mode == "OBJECT"

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        self.filepath = default_export_path(context, self.filename_ext)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class ExportFbxOperator(_ExportTakeOperator, bpy.types.Operator):
    bl_idname = "freeclipmocap.export_fbx"
    bl_label = "Export FBX"
    bl_description = "Export the current take with its armature (no mesh) as FBX, scaled like a Mixamo clip"
    filename_ext = ".fbx"
    filter_glob: bpy.props.StringProperty(default="*.fbx", options={"HIDDEN"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        export_fbx(context, context.scene.freeclipmocap.target_armature, Path(self.filepath))
        self.report({"INFO"}, f"Exported {self.filepath}")
        return {"FINISHED"}


class ExportBvhOperator(_ExportTakeOperator, bpy.types.Operator):
    bl_idname = "freeclipmocap.export_bvh"
    bl_label = "Export BVH"
    bl_description = "Export the current take over the scene's frame range as BVH (Y-up, metres)"
    filename_ext = ".bvh"
    filter_glob: bpy.props.StringProperty(default="*.bvh", options={"HIDDEN"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        export_bvh(context, context.scene.freeclipmocap.target_armature, Path(self.filepath))
        self.report({"INFO"}, f"Exported {self.filepath}")
        return {"FINISHED"}


register, unregister = bpy.utils.register_classes_factory((ExportFbxOperator, ExportBvhOperator))
