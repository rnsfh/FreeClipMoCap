import bpy
import numpy as np

from ..solver.rig_rest import RigRest


def rig_rest_from_armature(armature_object: bpy.types.Object) -> RigRest:
    ordered_bones: list[bpy.types.Bone] = []

    def visit(bone: bpy.types.Bone) -> None:
        ordered_bones.append(bone)
        for child in bone.children:
            visit(child)

    for bone in armature_object.data.bones:
        if bone.parent is None:
            visit(bone)
    index_by_name = {bone.name: index for index, bone in enumerate(ordered_bones)}
    return RigRest(
        bone_names=tuple(bone.name for bone in ordered_bones),
        parent_indices=tuple(index_by_name[bone.parent.name] if bone.parent else -1 for bone in ordered_bones),
        rest_rotations=np.array([np.array(bone.matrix_local.to_3x3()) for bone in ordered_bones], dtype=np.float64),
        heads=np.array([tuple(bone.head_local) for bone in ordered_bones], dtype=np.float64),
        tails=np.array([tuple(bone.tail_local) for bone in ordered_bones], dtype=np.float64),
    )
