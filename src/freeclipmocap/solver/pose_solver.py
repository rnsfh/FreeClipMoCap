import warnings as warnings_module
from dataclasses import dataclass, field

import numpy as np

from ..capture.landmark_cache import LandmarkCache
from ..constants.bone_roles import Role
from ..rig.bone_map import BoneMap
from . import gap_filling, grounding, one_euro_filter, rotations
from .finger_limits import FingerLimiter
from .rig_rest import RigRest
from .tracking_table import (
    Direction,
    FootRule,
    LandmarkPoint,
    PointSet,
    RigAxis,
    RigPoint,
    RoleRule,
    SolveMode,
    TrackingTable,
)

_MEDIAPIPE_RIGHT = np.array([-1.0, 0.0, 0.0])
_MEDIAPIPE_UP = np.array([0.0, -1.0, 0.0])
_BEND_WEIGHT_START = 0.25
_BEND_WEIGHT_FULL = 0.5
_DEGENERATE_REFERENCE = 1e-3
_GLITCH_WINDOW_FRAMES = 7
_GLITCH_COSINE = np.cos(np.radians(100.0))
_FOOT_IMAGE_SMOOTHING = 0.5
_LONGEST_HAND_BLEND_SECONDS = 1.0
_HAND_EASE_SECONDS = 0.25


class SolveError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class SolveSettings:
    smoothing: float
    root_motion: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class SolvedTake:
    times_seconds: np.ndarray
    rotations_by_bone: dict[str, np.ndarray]
    warnings: tuple[str, ...]
    camera_tilt_degrees: float
    locations_by_bone: dict[str, np.ndarray] = field(default_factory=dict)
    foot_bones: tuple[str, ...] = ()

    @property
    def frame_count(self) -> int:
        return len(self.times_seconds)

    def resampled(self, frame_rate: float) -> "SolvedTake":
        output_times = np.arange(0.0, self.times_seconds[-1] + 1e-9, 1.0 / frame_rate)
        if len(self.times_seconds) == 1:
            return SolvedTake(
                times_seconds=output_times,
                rotations_by_bone={
                    bone_name: np.repeat(quaternions, len(output_times), axis=0)
                    for bone_name, quaternions in self.rotations_by_bone.items()
                },
                warnings=self.warnings,
                camera_tilt_degrees=self.camera_tilt_degrees,
                locations_by_bone={
                    bone_name: np.repeat(locations, len(output_times), axis=0)
                    for bone_name, locations in self.locations_by_bone.items()
                },
                foot_bones=self.foot_bones,
            )
        lower_indices = np.clip(np.searchsorted(self.times_seconds, output_times, side="right") - 1, 0, len(self.times_seconds) - 2)
        lower_times = self.times_seconds[lower_indices]
        upper_times = self.times_seconds[lower_indices + 1]
        fractions = np.clip((output_times - lower_times) / np.maximum(upper_times - lower_times, 1e-9), 0.0, 1.0)
        return SolvedTake(
            times_seconds=output_times,
            rotations_by_bone={
                bone_name: rotations.make_continuous(
                    rotations.slerp(quaternions[lower_indices], quaternions[lower_indices + 1], fractions)
                )
                for bone_name, quaternions in self.rotations_by_bone.items()
            },
            warnings=self.warnings,
            camera_tilt_degrees=self.camera_tilt_degrees,
            locations_by_bone={
                bone_name: locations[lower_indices] + fractions[:, None] * (locations[lower_indices + 1] - locations[lower_indices])
                for bone_name, locations in self.locations_by_bone.items()
            },
            foot_bones=self.foot_bones,
        )


class _Solver:
    def __init__(self, rig: RigRest, bone_map: BoneMap, point_sets: dict[PointSet, np.ndarray], axes: dict[RigAxis, np.ndarray]) -> None:
        self._rig = rig
        self._bone_map = bone_map
        self._point_sets = point_sets
        self._axes = axes

    def actor_vector(self, direction: Direction) -> np.ndarray | None:
        start = self._actor_point(direction.actor_start)
        end = self._actor_point(direction.actor_end)
        if start is None or end is None:
            return None
        return end - start

    def rig_vector(self, direction: Direction) -> np.ndarray | None:
        if direction.rig_axis is not None:
            tilt = np.radians(direction.rig_axis_tilt_up_degrees)
            return np.cos(tilt) * self._axes[direction.rig_axis] + np.sin(tilt) * self._axes[RigAxis.UP]
        if direction.rig_start is None or direction.rig_end is None:
            return None
        start = self._rig_point(direction.rig_start)
        end = self._rig_point(direction.rig_end)
        if start is None or end is None:
            return None
        return end - start

    def apply(self, rule: RoleRule, current: np.ndarray, rest_rotation: np.ndarray) -> np.ndarray | None:
        target = self.actor_vector(rule.direction)
        if target is None:
            return None
        swung = _swing(current, target)
        if rule.mode == SolveMode.SWING or rule.twist is None:
            return swung
        target_twist = self.actor_vector(rule.twist)
        rig_twist = self.rig_vector(rule.twist)
        rig_direction = rest_rotation[:, 1] if rule.mode == SolveMode.BENDING_LIMB else self.rig_vector(rule.direction)
        if target_twist is None or rig_twist is None or rig_direction is None:
            return swung
        framed, twist_strength = _frame(target, target_twist, rig_direction, rig_twist, rest_rotation)
        if rule.mode == SolveMode.BENDING_LIMB:
            weight = np.clip((twist_strength - _BEND_WEIGHT_START) / (_BEND_WEIGHT_FULL - _BEND_WEIGHT_START), 0.0, 1.0)
        else:
            weight = (twist_strength > _DEGENERATE_REFERENCE).astype(np.float64)
        return rotations.slerp_matrices(swung, framed, weight)

    def frame_from_rule(self, direction: Direction, twist: Direction, rest_rotation: np.ndarray) -> np.ndarray | None:
        target = self.actor_vector(direction)
        target_twist = self.actor_vector(twist)
        rig_direction = self.rig_vector(direction)
        rig_twist = self.rig_vector(twist)
        if target is None or target_twist is None or rig_direction is None or rig_twist is None:
            return None
        framed, _twist_strength = _frame(target, target_twist, rig_direction, rig_twist, rest_rotation)
        return framed

    def _actor_point(self, point: LandmarkPoint) -> np.ndarray | None:
        positions = self._point_sets.get(point.point_set)
        if positions is None:
            return None
        selected = positions[:, list(point.indices)]
        if np.isnan(selected).any():
            return None
        return selected.mean(axis=1)

    def _rig_point(self, point: RigPoint) -> np.ndarray | None:
        bone_names = [self._bone_map.bone_for(role) for role in point.roles]
        if any(bone_name is None for bone_name in bone_names):
            return None
        return np.mean([self._rig.heads[self._rig.index_of(str(bone_name))] for bone_name in bone_names], axis=0)


def solve(cache: LandmarkCache, rig: RigRest, bone_map: BoneMap, table: TrackingTable, settings: SolveSettings) -> SolvedTake:
    if cache.tracking_model != table.tracking_model:
        raise SolveError(f"The cache comes from {cache.tracking_model}; this solver table is for {table.tracking_model}.")
    missing_roles = bone_map.missing_required_roles()
    if missing_roles:
        raise SolveError(f"The bone map misses required roles: {', '.join(role.value for role in missing_roles)}.")

    timestamps_seconds = (cache.timestamps_milliseconds - cache.timestamps_milliseconds[0]).astype(np.float64) / 1000.0
    axes = _rig_axes(rig, bone_map)
    to_armature = _mediapipe_to_armature(axes)
    warnings: list[str] = []
    point_sets: dict[PointSet, np.ndarray] = {}
    if not cache.pose.found.any():
        raise SolveError("No body was found in any frame of the capture.")
    pose_world = cache.pose.world_landmarks.astype(np.float64)
    glitch_frames = _left_right_glitch_frames(pose_world, (table.hips_rule.twist, table.chest_twist))
    good_frames = cache.pose.found & ~glitch_frames
    visible = good_frames[:, None] & (cache.pose.visibility >= table.minimum_visibility)
    visible[:, visible.mean(axis=0) < table.minimum_visible_share] = False
    filled_pose = gap_filling.hold_last_good_per_point(pose_world, visible)
    point_sets[PointSet.POSE] = one_euro_filter.smooth(filled_pose, timestamps_seconds, settings.smoothing) @ to_armature.T
    for point_set, track, description in (
        (PointSet.LEFT_HAND, cache.left_hand, "left hand"),
        (PointSet.RIGHT_HAND, cache.right_hand, "right hand"),
    ):
        filled = gap_filling.blend_gaps(
            track.world_landmarks.astype(np.float64),
            track.found & good_frames,
            timestamps_seconds,
            _LONGEST_HAND_BLEND_SECONDS,
            _HAND_EASE_SECONDS,
        )
        if filled is None:
            warnings.append(f"The {description} was never tracked; its fingers are not animated.")
            continue
        smoothed = one_euro_filter.smooth(filled, timestamps_seconds, settings.smoothing)
        point_sets[point_set] = smoothed @ to_armature.T

    torso_up = _Solver(rig, bone_map, point_sets, axes).actor_vector(table.chest_direction)
    if torso_up is None:
        raise SolveError("The hips and shoulders were never clearly visible, so the body cannot be solved.")
    median_torso_up = rotations.normalized(np.median(rotations.normalized(torso_up), axis=0))
    leveling = rotations.rotation_between(median_torso_up, axes[RigAxis.UP])
    camera_tilt_degrees = float(np.degrees(np.arccos(np.clip(np.dot(median_torso_up, axes[RigAxis.UP]), -1.0, 1.0))))
    point_sets = {point_set: positions @ leveling.T for point_set, positions in point_sets.items()}

    solver = _Solver(rig, bone_map, point_sets, axes)
    role_by_bone = {bone_name: role for role, bone_name in bone_map.bones_by_role.items()}
    rules_by_role = {rule.role: rule for rule in table.role_rules}
    hand_rules_by_role = {hand_rule.hand_role: hand_rule for hand_rule in table.hand_rules}
    finger_rules_by_role = {
        finger_rule.role: finger_rule for hand_rule in table.hand_rules for finger_rule in hand_rule.finger_rules
    }
    mapped_spine_roles = [role for role in table.spine_roles if bone_map.bone_for(role) is not None]
    finger_limiter = FingerLimiter.for_rig(rig, bone_map)
    chest_delta = solver.frame_from_rule(table.chest_direction, table.chest_twist, np.eye(3))

    frame_count = len(timestamps_seconds)
    identity = np.broadcast_to(np.eye(3), (frame_count, 3, 3))
    posed: dict[int, np.ndarray] = {}
    hand_space: dict[int, np.ndarray] = {}
    hips_delta = identity
    rotations_by_bone: dict[str, np.ndarray] = {}
    locations_by_bone: dict[str, np.ndarray] = {}
    unseen_roles: list[Role] = []

    for index, bone_name in enumerate(rig.bone_names):
        parent_index = rig.parent_indices[index]
        rest_rotation = rig.rest_rotations[index]
        if parent_index < 0:
            parent_global = identity
            rest_local = rest_rotation
        else:
            parent_global = hand_space.get(parent_index, posed[parent_index])
            rest_local = rig.rest_rotations[parent_index].T @ rest_rotation
        current = parent_global @ rest_local
        role = role_by_bone.get(bone_name)
        desired: np.ndarray | None = None
        if role == Role.HIPS:
            desired = solver.apply(table.hips_rule, current, rest_rotation)
            if desired is not None:
                hips_delta = desired @ rest_rotation.T
        elif role in mapped_spine_roles and chest_delta is not None:
            fraction = (mapped_spine_roles.index(role) + 1) / len(mapped_spine_roles)
            desired = rotations.slerp_matrices(hips_delta, chest_delta, fraction) @ rest_rotation
        elif role in rules_by_role:
            desired = solver.apply(rules_by_role[role], current, rest_rotation)
            if desired is None:
                unseen_roles.append(role)
        elif role in finger_rules_by_role and parent_index in hand_space:
            desired = solver.apply(finger_rules_by_role[role], current, rest_rotation)

        if desired is None:
            desired = current
        else:
            local = rest_local.T @ np.swapaxes(parent_global, -1, -2) @ desired
            if role in finger_rules_by_role:
                local = finger_limiter.limited(index, local)
                desired = parent_global @ rest_local @ local
            rotations_by_bone[bone_name] = _as_quaternions(local)
        posed[index] = desired

        if role in hand_rules_by_role:
            hand_rule = hand_rules_by_role[role]
            hand_frame = solver.frame_from_rule(hand_rule.direction, hand_rule.twist, rest_rotation)
            if hand_frame is not None:
                hand_space[index] = hand_frame
        elif parent_index in hand_space:
            hand_space[index] = desired

    if unseen_roles:
        warnings.append(
            "Too rarely visible in the video, so not animated: " + ", ".join(role.value for role in unseen_roles) + "."
        )
    legs = _leg_chains(rig, bone_map, table.foot_rules, rotations_by_bone)
    hips_name = str(bone_map.bone_for(Role.HIPS))
    if legs is None:
        warnings.append("The legs were not visible enough to plant the feet; the hips stay at their rest height.")
    else:
        hips_index = rig.index_of(hips_name)
        contacts = _foot_contacts(cache, table, visible, point_sets, timestamps_seconds, axes[RigAxis.UP])
        grounded = grounding.ground(
            rig,
            posed,
            hips_index,
            legs,
            contacts,
            axes[RigAxis.UP],
            axes[RigAxis.FRONT],
            timestamps_seconds,
            settings.smoothing,
            settings.root_motion,
        )
        posed.update(grounded.leg_rotations)
        for leg in legs:
            for index in (leg.up_leg, leg.leg, leg.foot):
                rotations_by_bone[rig.bone_names[index]] = _as_quaternions(_local_rotation(rig, index, posed))
        locations_by_bone[hips_name] = _hips_location(rig, hips_index, posed, grounded.hips_translation)
        if not contacts.any():
            warnings.append("No planted foot was found, so the lowest foot is kept on the floor.")
    glitch_count = int(glitch_frames.sum())
    if glitch_count:
        warnings.append(f"{glitch_count} frame(s) with left and right mixed up by the tracker were skipped.")
    return SolvedTake(
        times_seconds=timestamps_seconds,
        rotations_by_bone=rotations_by_bone,
        warnings=tuple(warnings),
        camera_tilt_degrees=camera_tilt_degrees,
        locations_by_bone=locations_by_bone,
        foot_bones=tuple(rig.bone_names[leg.foot] for leg in legs) if legs is not None else (),
    )


def _local_rotation(rig: RigRest, index: int, globals_by_index: dict[int, np.ndarray]) -> np.ndarray:
    parent_index = rig.parent_indices[index]
    rest_rotation = rig.rest_rotations[index]
    if parent_index < 0:
        return rest_rotation.T @ globals_by_index[index]
    rest_local = rig.rest_rotations[parent_index].T @ rest_rotation
    return rest_local.T @ np.swapaxes(globals_by_index[parent_index], -1, -2) @ globals_by_index[index]


def _as_quaternions(local_rotations: np.ndarray) -> np.ndarray:
    return rotations.make_continuous(rotations.quaternions_from_matrices(local_rotations))


def _leg_chains(
    rig: RigRest, bone_map: BoneMap, foot_rules: tuple[FootRule, ...], solved: dict[str, np.ndarray]
) -> tuple[grounding.LegChain, ...] | None:
    legs: list[grounding.LegChain] = []
    for foot_rule in foot_rules:
        names = [bone_map.bone_for(role) for role in (foot_rule.up_leg_role, foot_rule.leg_role, foot_rule.foot_role)]
        if any(name is None or name not in solved for name in names):
            return None
        up_leg, leg, foot = (rig.index_of(str(name)) for name in names)
        toe_name = bone_map.bone_for(foot_rule.toe_role)
        if toe_name is None:
            ground_points = ((foot, False), (foot, True))
        else:
            toe = rig.index_of(toe_name)
            ground_points = ((foot, False), (toe, False), (toe, True))
        legs.append(grounding.LegChain(up_leg, leg, foot, ground_points))
    return tuple(legs) if legs else None


def _foot_contacts(
    cache: LandmarkCache,
    table: TrackingTable,
    visible: np.ndarray,
    point_sets: dict[PointSet, np.ndarray],
    timestamps_seconds: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    pixels = cache.pose.image_landmarks[..., :2].astype(np.float64) * np.array([cache.video.width, cache.video.height])
    pixels = one_euro_filter.smooth(
        gap_filling.hold_last_good_per_point(pixels, visible), timestamps_seconds, _FOOT_IMAGE_SMOOTHING
    )
    world = point_sets[PointSet.POSE]

    def center(points: np.ndarray, landmark_point: LandmarkPoint) -> np.ndarray:
        return points[:, list(landmark_point.indices)].mean(axis=1)

    torso = table.chest_direction
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore", RuntimeWarning)
        torso_pixels = float(np.nanmedian(np.linalg.norm(center(pixels, torso.actor_end) - center(pixels, torso.actor_start), axis=-1)))
        torso_length = float(np.nanmedian(np.linalg.norm(center(world, torso.actor_end) - center(world, torso.actor_start), axis=-1)))
    return grounding.detect_contacts(
        np.stack([center(pixels, rule.contact_point) for rule in table.foot_rules]),
        np.stack([visible[:, list(rule.contact_point.indices)].all(axis=1) for rule in table.foot_rules]),
        np.stack([center(world, rule.contact_point) @ up for rule in table.foot_rules]),
        torso_pixels,
        torso_length,
        timestamps_seconds,
    )


def _hips_location(rig: RigRest, hips_index: int, posed: dict[int, np.ndarray], translation: np.ndarray) -> np.ndarray:
    parent_index = rig.parent_indices[hips_index]
    if parent_index >= 0:
        parent_turn = posed[parent_index] @ rig.rest_rotations[parent_index].T
        translation = np.einsum("fji,fj->fi", parent_turn, translation)
    return translation @ rig.rest_rotations[hips_index]


def _left_right_glitch_frames(pose_world: np.ndarray, side_lines: tuple[Direction | None, ...]) -> np.ndarray:
    glitches = np.zeros(len(pose_world), dtype=bool)
    for side_line in side_lines:
        if side_line is None:
            continue
        start = pose_world[:, list(side_line.actor_start.indices)].mean(axis=1)
        end = pose_world[:, list(side_line.actor_end.indices)].mean(axis=1)
        directions = rotations.normalized(end - start)
        padded = np.pad(directions, ((_GLITCH_WINDOW_FRAMES, _GLITCH_WINDOW_FRAMES), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * _GLITCH_WINDOW_FRAMES + 1, axis=0)
        with warnings_module.catch_warnings():
            warnings_module.simplefilter("ignore", RuntimeWarning)
            neighbourhood = rotations.normalized(np.nanmedian(windows, axis=-1))
        cosine = np.sum(directions * neighbourhood, axis=-1)
        glitches |= cosine < _GLITCH_COSINE
    return glitches


def _swing(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return rotations.rotation_between(current[..., :, 1], target) @ current


def _frame(
    target: np.ndarray,
    target_twist: np.ndarray,
    rig_direction: np.ndarray,
    rig_twist: np.ndarray,
    rest_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_frame, perpendicular_lengths = rotations.frame_from_directions(target, target_twist)
    rig_frame, _rig_lengths = rotations.frame_from_directions(rig_direction, rig_twist)
    twist_strength = perpendicular_lengths / np.maximum(np.linalg.norm(target_twist, axis=-1), 1e-9)
    return target_frame @ rig_frame.T @ rest_rotation, twist_strength


def _rig_axes(rig: RigRest, bone_map: BoneMap) -> dict[RigAxis, np.ndarray]:
    def head(role: Role) -> np.ndarray:
        return rig.heads[rig.index_of(str(bone_map.bone_for(role)))]

    right = rotations.normalized(head(Role.RIGHT_UP_LEG) - head(Role.LEFT_UP_LEG))
    up_direction = head(Role.HEAD) - head(Role.HIPS)
    up = rotations.normalized(up_direction - np.dot(up_direction, right) * right)
    back = np.cross(right, up)
    return {RigAxis.RIGHT: right, RigAxis.UP: up, RigAxis.BACK: back, RigAxis.FRONT: -back}


def _mediapipe_to_armature(axes: dict[RigAxis, np.ndarray]) -> np.ndarray:
    rig_basis = np.stack((axes[RigAxis.RIGHT], axes[RigAxis.UP], axes[RigAxis.BACK]), axis=-1)
    mediapipe_basis = np.stack((_MEDIAPIPE_RIGHT, _MEDIAPIPE_UP, np.cross(_MEDIAPIPE_RIGHT, _MEDIAPIPE_UP)), axis=-1)
    return rig_basis @ mediapipe_basis.T
