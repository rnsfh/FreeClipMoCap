from dataclasses import dataclass
from enum import Enum

from ..constants.bone_roles import Role


class PointSet(Enum):
    POSE = "pose"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"


class RigAxis(Enum):
    RIGHT = "right"
    UP = "up"
    FRONT = "front"
    BACK = "back"


class SolveMode(Enum):
    SWING = "swing"
    FRAME = "frame"
    BENDING_LIMB = "bending_limb"


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    point_set: PointSet
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RigPoint:
    roles: tuple[Role, ...]


@dataclass(frozen=True, slots=True)
class Direction:
    actor_start: LandmarkPoint
    actor_end: LandmarkPoint
    rig_start: RigPoint | None = None
    rig_end: RigPoint | None = None
    rig_axis: RigAxis | None = None
    rig_axis_tilt_up_degrees: float = 0.0


@dataclass(frozen=True, slots=True)
class RoleRule:
    role: Role
    mode: SolveMode
    direction: Direction
    twist: Direction | None = None


@dataclass(frozen=True, slots=True)
class HandRule:
    hand_role: Role
    point_set: PointSet
    direction: Direction
    twist: Direction
    finger_rules: tuple[RoleRule, ...]


@dataclass(frozen=True, slots=True)
class FootRule:
    up_leg_role: Role
    leg_role: Role
    foot_role: Role
    toe_role: Role
    contact_point: LandmarkPoint


@dataclass(frozen=True, slots=True)
class TrackingTable:
    tracking_model: str
    minimum_visibility: float
    minimum_visible_share: float
    hips_rule: RoleRule
    chest_direction: Direction
    chest_twist: Direction
    spine_roles: tuple[Role, ...]
    role_rules: tuple[RoleRule, ...]
    hand_rules: tuple[HandRule, ...]
    foot_rules: tuple[FootRule, ...]
