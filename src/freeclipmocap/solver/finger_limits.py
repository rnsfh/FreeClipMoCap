from dataclasses import dataclass

import numpy as np

from ..constants.bone_roles import Role
from ..rig.bone_map import BoneMap
from . import rotations
from .rig_rest import RigRest

_FINGER_NAMES = ("Index", "Middle", "Ring", "Pinky")


@dataclass(frozen=True, slots=True)
class JointLimit:
    curl_degrees: tuple[float, float]
    spread_degrees: tuple[float, float]
    twist_degrees: tuple[float, float]


_FINGER_JOINT_LIMITS = (
    JointLimit((-30.0, 95.0), (-20.0, 20.0), (-10.0, 10.0)),
    JointLimit((-5.0, 110.0), (-5.0, 5.0), (-5.0, 5.0)),
    JointLimit((-10.0, 90.0), (-5.0, 5.0), (-5.0, 5.0)),
)
_THUMB_JOINT_LIMITS = (
    JointLimit((-60.0, 60.0), (-60.0, 60.0), (-45.0, 45.0)),
    JointLimit((-20.0, 70.0), (-15.0, 15.0), (-20.0, 20.0)),
    JointLimit((-20.0, 90.0), (-5.0, 5.0), (-10.0, 10.0)),
)


@dataclass(frozen=True, slots=True)
class _LimitedJoint:
    curl_axis: np.ndarray
    spread_axis: np.ndarray
    limit: JointLimit


@dataclass(frozen=True, slots=True)
class FingerLimiter:
    joints_by_bone: dict[int, _LimitedJoint]

    @classmethod
    def for_rig(cls, rig: RigRest, bone_map: BoneMap) -> "FingerLimiter":
        joints_by_bone: dict[int, _LimitedJoint] = {}
        for side in ("Left", "Right"):
            palm_side = _palm_side(rig, bone_map, side)
            if palm_side is None:
                continue
            for finger, limits in (("Thumb", _THUMB_JOINT_LIMITS), *((name, _FINGER_JOINT_LIMITS) for name in _FINGER_NAMES)):
                for joint, limit in enumerate(limits, start=1):
                    bone_name = bone_map.bone_for(Role(f"{side}Hand{finger}{joint}"))
                    if bone_name is None:
                        continue
                    index = rig.index_of(bone_name)
                    rest_rotation = rig.rest_rotations[index]
                    direction = rest_rotation[:, 1]
                    spread_axis = rotations.normalized(palm_side - np.dot(palm_side, direction) * direction)
                    curl_axis = rotations.normalized(np.cross(direction, palm_side))
                    joints_by_bone[index] = _LimitedJoint(rest_rotation.T @ curl_axis, rest_rotation.T @ spread_axis, limit)
        return cls(joints_by_bone)

    def limited(self, bone_index: int, local_rotations: np.ndarray) -> np.ndarray:
        joint = self.joints_by_bone.get(bone_index)
        if joint is None:
            return local_rotations
        quaternions = rotations.quaternions_from_matrices(local_rotations)
        twist = 2.0 * np.arctan2(quaternions[:, 2], quaternions[:, 0])
        twist = np.arctan2(np.sin(twist), np.cos(twist))
        swing = rotations.log_matrices(local_rotations @ np.swapaxes(_about_y(twist), -1, -2))
        curl = np.clip(swing @ joint.curl_axis, *np.radians(joint.limit.curl_degrees))
        spread = np.clip(swing @ joint.spread_axis, *np.radians(joint.limit.spread_degrees))
        limited_swing = rotations.exp_vectors(curl[:, None] * joint.curl_axis + spread[:, None] * joint.spread_axis)
        return limited_swing @ _about_y(np.clip(twist, *np.radians(joint.limit.twist_degrees)))


def _palm_side(rig: RigRest, bone_map: BoneMap, side: str) -> np.ndarray | None:
    names = [bone_map.bone_for(Role(f"{side}{part}")) for part in ("Hand", "HandIndex1", "HandPinky1", "HandThumb3")]
    if any(name is None for name in names):
        return None
    hand, index, pinky, thumb = (rig.index_of(str(name)) for name in names)
    wrist = rig.heads[hand]
    normal = rotations.normalized(np.cross(rig.heads[index] - wrist, rig.heads[pinky] - wrist))
    thumb_side = float(np.dot(rig.tails[thumb] - wrist, normal))
    if abs(thumb_side) < 1e-6:
        return None
    return normal if thumb_side > 0.0 else -normal


def _about_y(angles: np.ndarray) -> np.ndarray:
    cosines = np.cos(angles)
    sines = np.sin(angles)
    zeros = np.zeros_like(angles)
    ones = np.ones_like(angles)
    return np.stack(
        (
            np.stack((cosines, zeros, sines), axis=-1),
            np.stack((zeros, ones, zeros), axis=-1),
            np.stack((-sines, zeros, cosines), axis=-1),
        ),
        axis=-2,
    )
