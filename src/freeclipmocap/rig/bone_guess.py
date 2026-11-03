import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from ..constants.bone_roles import Role
from .bone_map import BoneMap
from .bone_map_preset import BoneMapPreset, without_namespace

_SIDE_TOKENS = {"l": "Left", "left": "Left", "r": "Right", "right": "Right"}
_IGNORED_TOKENS = {"end", "tip", "nub", "twist", "roll", "ik", "pole", "target", "helper", "metacarpal", "top"}
_FINGER_PREFIXES = ("", "hand", "finger", "f")
_FINGER_NAMES = {
    "thumb": "Thumb",
    "index": "Index",
    "middle": "Middle",
    "ring": "Ring",
    "pinky": "Pinky",
    "pinkie": "Pinky",
    "little": "Pinky",
}


class _Kind(Enum):
    HIPS = "Hips"
    SPINE = "Spine"
    NECK = "Neck"
    HEAD = "Head"
    SHOULDER = "Shoulder"
    ARM = "Arm"
    FORE_ARM = "ForeArm"
    HAND = "Hand"
    UP_LEG = "UpLeg"
    LEG = "Leg"
    FOOT = "Foot"
    TOE_BASE = "ToeBase"


_KINDS_BY_NAME = {
    "hips": _Kind.HIPS,
    "hip": _Kind.HIPS,
    "pelvis": _Kind.HIPS,
    "spine": _Kind.SPINE,
    "chest": _Kind.SPINE,
    "upperchest": _Kind.SPINE,
    "torso": _Kind.SPINE,
    "neck": _Kind.NECK,
    "head": _Kind.HEAD,
    "shoulder": _Kind.SHOULDER,
    "clavicle": _Kind.SHOULDER,
    "collar": _Kind.SHOULDER,
    "collarbone": _Kind.SHOULDER,
    "arm": _Kind.ARM,
    "upperarm": _Kind.ARM,
    "uparm": _Kind.ARM,
    "forearm": _Kind.FORE_ARM,
    "lowerarm": _Kind.FORE_ARM,
    "elbow": _Kind.FORE_ARM,
    "hand": _Kind.HAND,
    "wrist": _Kind.HAND,
    "upleg": _Kind.UP_LEG,
    "thigh": _Kind.UP_LEG,
    "upperleg": _Kind.UP_LEG,
    "leg": _Kind.LEG,
    "lowerleg": _Kind.LEG,
    "calf": _Kind.LEG,
    "shin": _Kind.LEG,
    "knee": _Kind.LEG,
    "foot": _Kind.FOOT,
    "ankle": _Kind.FOOT,
    "toe": _Kind.TOE_BASE,
    "toes": _Kind.TOE_BASE,
    "toebase": _Kind.TOE_BASE,
    "ball": _Kind.TOE_BASE,
}
_CENTER_KINDS = {_Kind.HIPS, _Kind.SPINE, _Kind.NECK, _Kind.HEAD}
_SPINE_ROLES = (Role.SPINE, Role.SPINE_1, Role.SPINE_2)


@dataclass(frozen=True, slots=True)
class BoneGuess:
    bone_map: BoneMap
    preset_name: str | None


@dataclass(frozen=True, slots=True)
class _BoneDescription:
    name: str
    depth: int
    side: str | None
    kind: _Kind | None
    finger: str | None


def guess_bone_map(parents_by_bone: Mapping[str, str | None], presets: Sequence[BoneMapPreset]) -> BoneGuess:
    best_preset: BoneMapPreset | None = None
    best_preset_map = BoneMap(bones_by_role={})
    for preset in presets:
        preset_map = preset.matched_to(parents_by_bone.keys())
        if preset_map.missing_required_roles():
            continue
        if len(preset_map.bones_by_role) > len(best_preset_map.bones_by_role):
            best_preset = preset
            best_preset_map = preset_map
    name_map = guess_from_names(parents_by_bone)
    return BoneGuess(
        bone_map=best_preset_map.merged_with(name_map),
        preset_name=best_preset.name if best_preset is not None else None,
    )


def guess_from_names(parents_by_bone: Mapping[str, str | None]) -> BoneMap:
    descriptions = sorted(
        (_describe(name, _depth(name, parents_by_bone)) for name in parents_by_bone),
        key=lambda description: (description.depth, description.name),
    )
    bones_by_role: dict[Role, str] = {}

    spine_bones = [description.name for description in descriptions if description.kind == _Kind.SPINE]
    for role, bone_name in zip(_SPINE_ROLES, _pick_spine_bones(spine_bones)):
        bones_by_role[role] = bone_name

    for description in descriptions:
        if description.kind is None or description.kind == _Kind.SPINE:
            continue
        if description.kind in _CENTER_KINDS:
            if description.side is not None:
                continue
            role = Role(description.kind.value)
        elif description.side is None:
            continue
        else:
            role = Role(f"{description.side}{description.kind.value}")
        bones_by_role.setdefault(role, description.name)

    joints_by_finger: dict[tuple[str, str], list[str]] = {}
    for description in descriptions:
        if description.finger is not None and description.side is not None:
            joints_by_finger.setdefault((description.side, description.finger), []).append(description.name)
    for (side, finger), bone_names in joints_by_finger.items():
        for joint, bone_name in enumerate(bone_names[:3], start=1):
            bones_by_role[Role(f"{side}Hand{finger}{joint}")] = bone_name

    return BoneMap(bones_by_role=bones_by_role)


def _pick_spine_bones(spine_bones: list[str]) -> list[str]:
    if len(spine_bones) <= 3:
        return spine_bones
    return [spine_bones[0], spine_bones[len(spine_bones) // 2], spine_bones[-1]]


def _depth(bone_name: str, parents_by_bone: Mapping[str, str | None]) -> int:
    depth = 0
    parent = parents_by_bone.get(bone_name)
    while parent is not None and depth < len(parents_by_bone):
        depth += 1
        parent = parents_by_bone.get(parent)
    return depth


def _tokens(bone_name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", without_namespace(bone_name))
    return [token.lower() for token in re.split(r"[\s._\-]+", spaced) if token]


def _describe(bone_name: str, depth: int) -> _BoneDescription:
    tokens = _tokens(bone_name)
    side = next((_SIDE_TOKENS[token] for token in tokens if token in _SIDE_TOKENS), None)
    remaining = [token for token in tokens if token not in _SIDE_TOKENS]
    if any(token in _IGNORED_TOKENS for token in remaining):
        return _BoneDescription(bone_name, depth, side, None, None)
    letters = re.sub(r"\d+", "", "".join(remaining))
    for prefix in _FINGER_PREFIXES:
        if letters.startswith(prefix) and letters[len(prefix):] in _FINGER_NAMES:
            return _BoneDescription(bone_name, depth, side, None, _FINGER_NAMES[letters[len(prefix):]])
    return _BoneDescription(bone_name, depth, side, _KINDS_BY_NAME.get(letters), None)
