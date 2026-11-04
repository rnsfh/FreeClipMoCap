import bpy

from ..animation.ik_controls import (
    IkControlsError,
    add_ik_controls,
    has_ik_controls,
    key_ik_targets_from_fk,
    remove_ik_controls,
)
from .solve_operator import leave_edit_mode


def _armature_with_take(context: bpy.types.Context) -> bpy.types.Object | None:
    armature_object = context.scene.freeclipmocap.target_armature
    if armature_object is None or armature_object.animation_data is None or armature_object.animation_data.action is None:
        return None
    return armature_object


class AddIkControlsOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.add_ik_controls"
    bl_label = "Add IK Controls"
    bl_description = (
        "Add a target empty (foot or hand) and a pole empty (knee or elbow direction) for each mapped leg and arm, "
        "keyed to follow the current take, with IK constraints driven by one IK/FK slider per limb. At 1 the limb "
        "reaches for its target, at 0 it plays the solved FK keys; both look the same until a target is moved"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        armature_object = _armature_with_take(context)
        return armature_object is not None and not has_ik_controls(armature_object)

    def execute(self, context: bpy.types.Context) -> set[str]:
        armature_object = _armature_with_take(context)
        if not leave_edit_mode(context, armature_object):
            self.report({"ERROR"}, f"Leave Edit Mode on {armature_object.name} first (Tab).")
            return {"CANCELLED"}
        try:
            added = add_ik_controls(context.scene, armature_object)
        except IkControlsError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"IK controls added: {', '.join(added)}. Move the empties to change the pose while IK is on.")
        return {"FINISHED"}


class MatchIkToFkOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.match_ik_to_fk"
    bl_label = "Match IK to FK"
    bl_description = (
        "Key the IK targets and poles again so they follow the FK keys of the current take. "
        "Replaces any moves made to the target and pole empties"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        armature_object = _armature_with_take(context)
        return armature_object is not None and has_ik_controls(armature_object)

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context: bpy.types.Context) -> set[str]:
        armature_object = _armature_with_take(context)
        if not leave_edit_mode(context, armature_object):
            self.report({"ERROR"}, f"Leave Edit Mode on {armature_object.name} first (Tab).")
            return {"CANCELLED"}
        key_ik_targets_from_fk(context.scene, armature_object)
        self.report({"INFO"}, "IK targets and poles keyed to follow the FK keys again.")
        return {"FINISHED"}


class RemoveIkControlsOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.remove_ik_controls"
    bl_label = "Remove IK Controls"
    bl_description = "Delete the IK target and pole empties, their constraints and the IK/FK sliders"
    bl_options = {"REGISTER", "UNDO"}

    keep_pose: bpy.props.BoolProperty(
        name="Keep the Pose",
        description="Bake what the limbs do now (IK where a slider is up) into their FK keys before removing the controls",
        default=True,
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        armature_object = _armature_with_take(context)
        return armature_object is not None and has_ik_controls(armature_object)

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context: bpy.types.Context) -> set[str]:
        armature_object = _armature_with_take(context)
        if not leave_edit_mode(context, armature_object):
            self.report({"ERROR"}, f"Leave Edit Mode on {armature_object.name} first (Tab).")
            return {"CANCELLED"}
        properties = context.scene.freeclipmocap
        key_tolerance = properties.key_tolerance_degrees if properties.reduce_keys else None
        removed = remove_ik_controls(context.scene, armature_object, self.keep_pose, key_tolerance)
        kept = " The pose was baked into the FK keys." if self.keep_pose else ""
        self.report({"INFO"}, f"IK controls removed: {', '.join(removed)}.{kept}")
        return {"FINISHED"}


register, unregister = bpy.utils.register_classes_factory(
    (AddIkControlsOperator, MatchIkToFkOperator, RemoveIkControlsOperator)
)
