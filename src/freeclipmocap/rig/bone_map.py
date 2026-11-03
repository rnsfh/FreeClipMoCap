from collections.abc import Mapping
from dataclasses import dataclass

from ..constants.bone_roles import REQUIRED_ROLES, Role


@dataclass(frozen=True, slots=True)
class BoneMap:
    bones_by_role: Mapping[Role, str]

    def bone_for(self, role: Role) -> str | None:
        return self.bones_by_role.get(role)

    def missing_required_roles(self) -> list[Role]:
        return [role for role in REQUIRED_ROLES if role not in self.bones_by_role]

    def merged_with(self, fallback: "BoneMap") -> "BoneMap":
        used_bones = set(self.bones_by_role.values())
        merged = dict(self.bones_by_role)
        for role, bone_name in fallback.bones_by_role.items():
            if role not in merged and bone_name not in used_bones:
                merged[role] = bone_name
                used_bones.add(bone_name)
        return BoneMap(bones_by_role=merged)
