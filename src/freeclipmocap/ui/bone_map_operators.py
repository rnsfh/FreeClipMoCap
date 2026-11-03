import bpy

from ..rig.armature_bone_map import (
    available_presets,
    bone_parents,
    guess_and_write_bone_map,
    has_bone_map,
    read_bone_map,
    read_armature_rig_settings,
    user_presets_folder,
    write_bone_map,
    write_rig_settings,
)
from ..rig.bone_map import BoneMap
from ..rig.bone_map_preset import BoneMapPreset, PresetError, is_valid_preset_name, preset_path

_REPLACE_WARNING = "This replaces the current bone map, including your edits."


def target_armature(context: bpy.types.Context) -> bpy.types.Object | None:
    return context.scene.freeclipmocap.target_armature


def describe_bone_map(bone_map: BoneMap) -> str:
    missing = bone_map.missing_required_roles()
    text = f"{len(bone_map.bones_by_role)} roles mapped"
    if missing:
        text += f"; missing required: {', '.join(role.value for role in missing)}"
    return text + "."


class GuessBoneMapOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.guess_bone_map"
    bl_label = "Guess Bone Map"
    bl_description = "Guess which bone plays each role, from the presets and then from bone names and hierarchy"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return target_armature(context) is not None

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        armature_object = target_armature(context)
        if armature_object is not None and has_bone_map(armature_object):
            return context.window_manager.invoke_confirm(self, event, message=_REPLACE_WARNING)
        return self.execute(context)

    def execute(self, context: bpy.types.Context) -> set[str]:
        armature_object = target_armature(context)
        if armature_object is None:
            return {"CANCELLED"}
        guess = guess_and_write_bone_map(armature_object)
        source = f"preset {guess.preset_name} and bone names" if guess.preset_name else "bone names"
        self.report({"INFO"}, f"Guessed from {source}: {describe_bone_map(guess.bone_map)}")
        return {"FINISHED"}


class ApplyBoneMapPresetOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.apply_bone_map_preset"
    bl_label = "Apply Preset"
    bl_description = "Map the bones as the chosen preset says (a namespace prefix such as 'mixamorig:' is ignored)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return target_armature(context) is not None and bool(context.scene.freeclipmocap.bone_map_preset)

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        armature_object = target_armature(context)
        if armature_object is not None and has_bone_map(armature_object):
            return context.window_manager.invoke_confirm(self, event, message=_REPLACE_WARNING)
        return self.execute(context)

    def execute(self, context: bpy.types.Context) -> set[str]:
        armature_object = target_armature(context)
        preset_name = context.scene.freeclipmocap.bone_map_preset
        presets, _problems = available_presets()
        preset = next((preset for preset in presets if preset.name == preset_name), None)
        if armature_object is None or preset is None:
            self.report({"ERROR"}, f"Preset not found: {preset_name}")
            return {"CANCELLED"}
        bone_map = preset.matched_to(bone_parents(armature_object).keys())
        write_bone_map(armature_object, bone_map)
        write_rig_settings(armature_object, preset)
        self.report({"INFO"}, f"Applied preset {preset.name}: {describe_bone_map(bone_map)}")
        return {"FINISHED"}


class SaveBoneMapPresetOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.save_bone_map_preset"
    bl_label = "Save Preset"
    bl_description = "Save the current bone map as a preset in your FreeClipMocap user folder"

    preset_name: bpy.props.StringProperty(name="Name", default="My Rig")

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        armature_object = target_armature(context)
        return armature_object is not None and has_bone_map(armature_object)

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: bpy.types.Context) -> None:
        self.layout.prop(self, "preset_name")
        problem = self._name_problem()
        if problem:
            self.layout.label(text=problem, icon="ERROR")
        elif preset_path(user_presets_folder(), self.preset_name).exists():
            self.layout.label(text="A preset with this name exists and will be overwritten.", icon="INFO")

    def execute(self, context: bpy.types.Context) -> set[str]:
        armature_object = target_armature(context)
        problem = self._name_problem()
        if armature_object is None or problem:
            self.report({"ERROR"}, problem or "No armature picked.")
            return {"CANCELLED"}
        preset = BoneMapPreset(
            name=self.preset_name,
            bones_by_role=read_bone_map(armature_object).bones_by_role,
            is_built_in=False,
            rig_settings=read_armature_rig_settings(armature_object),
        )
        try:
            path = preset.save(user_presets_folder())
        except (OSError, PresetError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        context.scene.freeclipmocap.bone_map_preset = preset.name
        self.report({"INFO"}, f"Saved preset {preset.name}: {path}")
        return {"FINISHED"}

    def _name_problem(self) -> str | None:
        if not is_valid_preset_name(self.preset_name):
            return "Use letters, digits, spaces, '-' and '_' only."
        presets, _problems = available_presets()
        if any(preset.is_built_in and preset.name == self.preset_name for preset in presets):
            return "That is the name of a built-in preset."
        return None


def chosen_preset(context: bpy.types.Context) -> BoneMapPreset | None:
    preset_name = context.scene.freeclipmocap.bone_map_preset
    presets, _problems = available_presets()
    return next((preset for preset in presets if preset.name == preset_name), None)


class DeleteBoneMapPresetOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.delete_bone_map_preset"
    bl_label = "Delete Preset"
    bl_description = "Delete the chosen preset from your FreeClipMocap user folder (built-in presets cannot be deleted)"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        preset = chosen_preset(context)
        return preset is not None and not preset.is_built_in

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        preset = chosen_preset(context)
        name = preset.name if preset is not None else ""
        return context.window_manager.invoke_confirm(
            self, event, message=f"Delete the preset {name}? Its file is removed for good.", confirm_text="Delete"
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        preset = chosen_preset(context)
        if preset is None:
            return {"CANCELLED"}
        try:
            preset.delete()
        except (OSError, PresetError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        remaining_presets, _problems = available_presets()
        if remaining_presets:
            context.scene.freeclipmocap.bone_map_preset = remaining_presets[0].name
        self.report({"INFO"}, f"Deleted preset {preset.name}.")
        return {"FINISHED"}


_classes = (GuessBoneMapOperator, ApplyBoneMapPresetOperator, SaveBoneMapPresetOperator, DeleteBoneMapPresetOperator)


def register() -> None:
    for operator_class in _classes:
        bpy.utils.register_class(operator_class)


def unregister() -> None:
    for operator_class in reversed(_classes):
        bpy.utils.unregister_class(operator_class)
