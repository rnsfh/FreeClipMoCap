from dataclasses import dataclass

import numpy as np

from . import one_euro_filter, rotations
from .rig_rest import RigRest

_CONTACT_SPEED_TORSOS_PER_SECOND = 0.6
_CONTACT_LIFT_TORSOS = 0.12
_LONGEST_CONTACT_GAP_FRAMES = 2
_SHORTEST_CONTACT_FRAMES = 3
_LOCK_BLEND_FRAMES = 3
_IN_PLACE_WINDOW_SECONDS = 1.5
_SMALLEST_HEADING_SHARE = 0.05
_LARGEST_KNEE_CHANGE_DEGREES = 3.0


@dataclass(frozen=True, slots=True)
class LegChain:
    up_leg: int
    leg: int
    foot: int
    ground_points: tuple[tuple[int, bool], ...]


@dataclass(frozen=True, slots=True, eq=False)
class Grounding:
    hips_translation: np.ndarray
    leg_rotations: dict[int, np.ndarray]
    contacts: np.ndarray


def detect_contacts(
    foot_pixels: np.ndarray,
    foot_visible: np.ndarray,
    foot_heights: np.ndarray,
    torso_pixels: float,
    torso_length: float,
    timestamps_seconds: np.ndarray,
) -> np.ndarray:
    contacts = np.zeros(foot_visible.shape, dtype=bool)
    if len(timestamps_seconds) < 3 or torso_pixels <= 0.0 or torso_length <= 0.0:
        return contacts
    lowest = np.nanmin(foot_heights, axis=0)
    for foot in range(len(foot_pixels)):
        velocity = np.gradient(foot_pixels[foot], timestamps_seconds, axis=0)
        speed = np.linalg.norm(velocity, axis=-1) / torso_pixels
        lift = (foot_heights[foot] - lowest) / torso_length
        with np.errstate(invalid="ignore"):
            raw = foot_visible[foot] & (speed < _CONTACT_SPEED_TORSOS_PER_SECOND) & (lift < _CONTACT_LIFT_TORSOS)
        contacts[foot] = _cleaned(raw)
    return contacts


def joint_positions(
    rig: RigRest,
    globals_by_index: dict[int, np.ndarray],
    hips_index: int,
    hips_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(hips_translation)
    heads = np.empty((frame_count, len(rig.bone_names), 3))
    tails = np.empty_like(heads)
    for index, parent in enumerate(rig.parent_indices):
        if parent < 0:
            head = np.broadcast_to(rig.heads[index], (frame_count, 3)).copy()
        else:
            offset = rig.rest_rotations[parent].T @ (rig.heads[index] - rig.heads[parent])
            head = heads[:, parent] + globals_by_index[parent] @ offset
        if index == hips_index:
            head = head + hips_translation
        heads[:, index] = head
        tails[:, index] = head + globals_by_index[index] @ (rig.rest_rotations[index].T @ (rig.tails[index] - rig.heads[index]))
    return heads, tails


def ground(
    rig: RigRest,
    globals_by_index: dict[int, np.ndarray],
    hips_index: int,
    legs: tuple[LegChain, ...],
    contacts: np.ndarray,
    body_up: np.ndarray,
    front: np.ndarray,
    timestamps_seconds: np.ndarray,
    smoothing: float,
    root_motion: bool,
) -> Grounding:
    frame_count = len(timestamps_seconds)
    up = rotations.nearest_axis(body_up)
    weights = _lock_weights(contacts)
    flat_feet = _flattened_feet(rig, globals_by_index, legs, weights, up)
    globals_by_index = _with_turned_bones(rig, globals_by_index, flat_feet)
    rest_heights = [_point_heights(rig.heads[None], rig.tails[None], leg, up)[0] for leg in legs]
    floor = min(float(heights.min()) for heights in rest_heights)
    clearances = [heights - floor for heights in rest_heights]

    heads, tails = joint_positions(rig, globals_by_index, hips_index, np.zeros((frame_count, 3)))
    distances = np.stack(
        [(_point_heights(heads, tails, leg, up) - clearance).min(axis=-1) - floor for leg, clearance in zip(legs, clearances)]
    )

    lift = _hips_lift(distances, contacts)
    lift = one_euro_filter.smooth(lift[:, None], timestamps_seconds, smoothing)[:, 0]
    ankles = np.stack([heads[:, leg.foot] for leg in legs])
    travel = _travel(ankles, contacts, up)
    root_translation = travel + lift[:, None] * up

    root_heads, _root_tails = joint_positions(rig, globals_by_index, hips_index, root_translation)
    targets = _planted_targets(root_heads, legs, contacts, weights, distances + lift, up)
    root_translation = root_translation - _reach_drop(rig, root_heads, legs, targets, weights, up)[:, None] * up
    leg_rotations = _lock_planted_feet(rig, globals_by_index, hips_index, legs, targets, weights, root_translation, front)
    leg_rotations.update(flat_feet)
    hips_translation = root_translation
    if not root_motion and frame_count > 1:
        frame_seconds = max(float(np.median(np.diff(timestamps_seconds))), 1e-4)
        window = max(1, int(round(_IN_PLACE_WINDOW_SECONDS / frame_seconds)))
        stance = _moving_average(_stance(targets, contacts, travel, up), window)
        hips_translation = root_translation - (stance - stance[0])
    return Grounding(hips_translation=hips_translation, leg_rotations=leg_rotations, contacts=contacts)


def _point_heights(heads: np.ndarray, tails: np.ndarray, leg: LegChain, up: np.ndarray) -> np.ndarray:
    return np.stack([(tails if is_tail else heads)[:, bone] @ up for bone, is_tail in leg.ground_points], axis=-1)


def _hips_lift(distances: np.ndarray, contacts: np.ndarray) -> np.ndarray:
    any_contact = contacts.any(axis=0)
    if not any_contact.any():
        return -distances.min(axis=0)
    required = np.where(contacts, -distances, np.inf).min(axis=0)
    frames = np.arange(distances.shape[1])
    return np.interp(frames, frames[any_contact], required[any_contact])


def _travel(ankles: np.ndarray, contacts: np.ndarray, up: np.ndarray) -> np.ndarray:
    steps = _horizontal(ankles[:, 1:] - ankles[:, :-1], up)
    planted_both_frames = (contacts[:, 1:] & contacts[:, :-1]).astype(np.float64)
    planted_count = planted_both_frames.sum(axis=0)
    mean_step = (steps * planted_both_frames[..., None]).sum(axis=0) / np.maximum(planted_count, 1.0)[:, None]
    return np.concatenate((np.zeros((1, 3)), np.cumsum(-mean_step, axis=0)))


def _lock_weights(contacts: np.ndarray) -> np.ndarray:
    frame_count = contacts.shape[1]
    weights = np.zeros(contacts.shape)
    for leg_index, planted in enumerate(contacts):
        for start, stop in _runs(planted):
            steps_in = np.arange(stop - start, dtype=np.float64)
            blend_in = steps_in + 1 if start > 0 else np.inf
            blend_out = stop - start - steps_in if stop < frame_count else np.inf
            weights[leg_index, start:stop] = np.minimum(1.0, np.minimum(blend_in, blend_out) / (_LOCK_BLEND_FRAMES + 1))
    return weights


def _flattened_feet(
    rig: RigRest, globals_by_index: dict[int, np.ndarray], legs: tuple[LegChain, ...], weights: np.ndarray, up: np.ndarray
) -> dict[int, np.ndarray]:
    flat_feet: dict[int, np.ndarray] = {}
    for leg_index, leg in enumerate(legs):
        rest_rotation = rig.rest_rotations[leg.foot]
        tracked = globals_by_index[leg.foot]
        rest_heading = _horizontal(rig.tails[leg.foot] - rig.heads[leg.foot], up)
        if np.linalg.norm(rest_heading) < _SMALLEST_HEADING_SHARE * np.linalg.norm(rig.tails[leg.foot] - rig.heads[leg.foot]):
            flat_feet[leg.foot] = tracked
            continue
        heading = _horizontal(np.einsum("fij,jk,k->fi", tracked, rest_rotation.T, rest_heading), up)
        is_upright = np.linalg.norm(heading, axis=-1) < _SMALLEST_HEADING_SHARE * np.linalg.norm(rest_heading)
        heading[is_upright] = rest_heading
        flat = _turns_about(up, _signed_angles(rest_heading, heading, up)) @ rest_rotation
        flat_feet[leg.foot] = rotations.slerp_matrices(tracked, flat, weights[leg_index])
    return flat_feet


def _with_turned_bones(
    rig: RigRest, globals_by_index: dict[int, np.ndarray], turned: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    updated = dict(globals_by_index)
    turns: dict[int, np.ndarray] = {}
    for index, parent in enumerate(rig.parent_indices):
        if index in turned:
            turns[index] = turned[index] @ np.swapaxes(globals_by_index[index], -1, -2)
        elif parent in turns and index in globals_by_index:
            turns[index] = turns[parent]
        else:
            continue
        updated[index] = turns[index] @ globals_by_index[index]
    return updated


def _signed_angles(from_vector: np.ndarray, to_vectors: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.arctan2(np.cross(from_vector, to_vectors) @ axis, to_vectors @ from_vector)


def _turns_about(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    sine = np.sin(angles)[:, None, None]
    cosine = np.cos(angles)[:, None, None]
    return np.eye(3) + sine * cross + (1.0 - cosine) * (cross @ cross)


def _planted_targets(
    heads: np.ndarray,
    legs: tuple[LegChain, ...],
    contacts: np.ndarray,
    weights: np.ndarray,
    floor_distances: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    targets = np.empty((len(legs), len(heads), 3))
    for leg_index, leg in enumerate(legs):
        ankle = heads[:, leg.foot]
        target = ankle - floor_distances[leg_index][:, None] * up
        for start, stop in _runs(contacts[leg_index]):
            held = np.median(_horizontal(ankle[start:stop], up), axis=0)
            target[start:stop] = held + (target[start:stop] @ up)[:, None] * up
        targets[leg_index] = ankle + weights[leg_index][:, None] * (target - ankle)
    return targets


def _reach_drop(
    rig: RigRest, heads: np.ndarray, legs: tuple[LegChain, ...], targets: np.ndarray, weights: np.ndarray, up: np.ndarray
) -> np.ndarray:
    drop = np.zeros(len(heads))
    for leg_index, leg in enumerate(legs):
        reach = np.linalg.norm(rig.heads[leg.leg] - rig.heads[leg.up_leg]) + np.linalg.norm(rig.heads[leg.foot] - rig.heads[leg.leg])
        to_target = targets[leg_index] - heads[:, leg.up_leg]
        vertical = to_target @ up
        sideways_squared = np.sum(_horizontal(to_target, up) ** 2, axis=-1)
        reachable_vertical = -np.sqrt(np.maximum(reach**2 - sideways_squared, 0.0))
        needed = np.where((weights[leg_index] > 0.0) & (vertical < reachable_vertical), reachable_vertical - vertical, 0.0)
        drop = np.maximum(drop, needed)
    return drop


def _stance(targets: np.ndarray, contacts: np.ndarray, travel: np.ndarray, up: np.ndarray) -> np.ndarray:
    frames = np.arange(contacts.shape[1])
    spots: list[np.ndarray] = []
    for leg_index, planted in enumerate(contacts):
        runs = _runs(planted)
        if not runs:
            continue
        known_frames: list[int] = []
        known_spots: list[np.ndarray] = []
        for start, stop in runs:
            spot = np.median(_horizontal(targets[leg_index, start:stop], up), axis=0)
            known_frames += [start, stop - 1]
            known_spots += [spot, spot]
        known = np.array(known_spots)
        spots.append(np.stack([np.interp(frames, known_frames, known[:, axis]) for axis in range(3)], axis=-1))
    return np.mean(spots, axis=0) if spots else travel


def _lock_planted_feet(
    rig: RigRest,
    globals_by_index: dict[int, np.ndarray],
    hips_index: int,
    legs: tuple[LegChain, ...],
    targets: np.ndarray,
    weights: np.ndarray,
    root_translation: np.ndarray,
    front: np.ndarray,
) -> dict[int, np.ndarray]:
    heads, _tails = joint_positions(rig, globals_by_index, hips_index, root_translation)
    leg_rotations: dict[int, np.ndarray] = {}
    for leg_index, leg in enumerate(legs):
        thigh = globals_by_index[leg.up_leg]
        shin = globals_by_index[leg.leg]
        knee_front = thigh @ (rig.rest_rotations[leg.up_leg].T @ front)
        locked_thigh, locked_shin = _two_bone_ik(
            heads[:, leg.up_leg],
            heads[:, leg.leg],
            heads[:, leg.foot],
            targets[leg_index],
            thigh,
            shin,
            float(np.linalg.norm(rig.heads[leg.leg] - rig.heads[leg.up_leg])),
            float(np.linalg.norm(rig.heads[leg.foot] - rig.heads[leg.leg])),
            knee_front,
        )
        is_locked = (weights[leg_index] > 0.0)[:, None, None]
        leg_rotations[leg.up_leg] = np.where(is_locked, locked_thigh, thigh)
        leg_rotations[leg.leg] = np.where(is_locked, locked_shin, shin)
    return leg_rotations


def _two_bone_ik(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    target: np.ndarray,
    thigh: np.ndarray,
    shin: np.ndarray,
    thigh_length: float,
    shin_length: float,
    knee_front: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reach = target - hip
    tracked_bend = np.arccos(np.clip(np.sum(rotations.normalized(knee - hip) * rotations.normalized(ankle - knee), axis=-1), -1.0, 1.0))
    # Near a straight knee a few millimetres of reach cost many degrees of bend, so the lock may turn the knee only a
    # little away from the tracked bend; the foot then stays that little short of (or past) its target instead.
    largest_change = np.radians(_LARGEST_KNEE_CHANGE_DEGREES)
    shortest = _leg_reach(thigh_length, shin_length, np.minimum(tracked_bend + largest_change, np.pi))
    longest = _leg_reach(thigh_length, shin_length, np.maximum(tracked_bend - largest_change, 0.0))
    distance = np.clip(np.linalg.norm(reach, axis=-1), shortest, longest)
    along = rotations.normalized(reach)
    bend = knee - hip
    knee_side = bend - np.sum(bend * along, axis=-1, keepdims=True) * along
    front_side = knee_front - np.sum(knee_front * along, axis=-1, keepdims=True) * along
    is_straight = np.linalg.norm(knee_side, axis=-1, keepdims=True) < 1e-3 * thigh_length
    knee_side = rotations.normalized(np.where(is_straight, front_side, knee_side))
    cosine = np.clip((thigh_length**2 + distance**2 - shin_length**2) / (2.0 * thigh_length * distance), -1.0, 1.0)
    new_knee = hip + thigh_length * (cosine[:, None] * along + np.sqrt(1.0 - cosine**2)[:, None] * knee_side)
    new_ankle = hip + distance[:, None] * along
    thigh_turn = rotations.rotation_between(bend, new_knee - hip)
    turned_shin = np.einsum("fij,fj->fi", thigh_turn, ankle - knee)
    shin_turn = rotations.rotation_between(turned_shin, new_ankle - new_knee)
    return thigh_turn @ thigh, shin_turn @ thigh_turn @ shin


def _leg_reach(thigh_length: float, shin_length: float, bend: np.ndarray) -> np.ndarray:
    return np.sqrt(thigh_length**2 + shin_length**2 + 2.0 * thigh_length * shin_length * np.cos(bend))


def _horizontal(vectors: np.ndarray, up: np.ndarray) -> np.ndarray:
    return vectors - (vectors @ up)[..., None] * up


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    before = window // 2
    padded = np.pad(values, ((before, window - 1 - before), (0, 0)), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, window, axis=0).mean(axis=-1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(changes[0::2], changes[1::2])]


def _cleaned(raw: np.ndarray) -> np.ndarray:
    cleaned = raw.copy()
    for start, stop in _runs(~raw):
        if start > 0 and stop < len(raw) and stop - start <= _LONGEST_CONTACT_GAP_FRAMES:
            cleaned[start:stop] = True
    for start, stop in _runs(cleaned):
        if stop - start < _SHORTEST_CONTACT_FRAMES:
            cleaned[start:stop] = False
    return cleaned
