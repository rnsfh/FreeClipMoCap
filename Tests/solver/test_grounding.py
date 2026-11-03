import unittest

import numpy as np

from bl_ext.user_default.freeclipmocap.constants import pose_landmarks
from bl_ext.user_default.freeclipmocap.solver import grounding, rotations
from bl_ext.user_default.freeclipmocap.solver.mediapipe_table import MEDIAPIPE_TABLE
from bl_ext.user_default.freeclipmocap.solver.pose_solver import SolveSettings, SolvedTake, solve
from bl_ext.user_default.freeclipmocap.solver.rig_rest import RigRest

from .synthetic_actor import make_cache, make_rig, rest_landmarks, rotation_about

_NO_SMOOTHING = SolveSettings(smoothing=0.0)
_FOOT_POINTS = {
    "Left": [pose_landmarks.LEFT_ANKLE, pose_landmarks.LEFT_HEEL, pose_landmarks.LEFT_FOOT_INDEX],
    "Right": [pose_landmarks.RIGHT_ANKLE, pose_landmarks.RIGHT_HEEL, pose_landmarks.RIGHT_FOOT_INDEX],
}
_HIPS = "mixamorig:Hips"


def solved_joint_positions(rig: RigRest, take: SolvedTake) -> np.ndarray:
    frame_count = take.frame_count
    globals_by_index: dict[int, np.ndarray] = {}
    for index, bone_name in enumerate(rig.bone_names):
        parent = rig.parent_indices[index]
        if bone_name in take.rotations_by_bone:
            local = rotations.matrices_from_quaternions(take.rotations_by_bone[bone_name])
        else:
            local = np.broadcast_to(np.eye(3), (frame_count, 3, 3))
        if parent < 0:
            globals_by_index[index] = rig.rest_rotations[index] @ local
        else:
            rest_local = rig.rest_rotations[parent].T @ rig.rest_rotations[index]
            globals_by_index[index] = globals_by_index[parent] @ rest_local @ local
    hips_index = rig.index_of(_HIPS)
    translation = take.locations_by_bone[_HIPS] @ rig.rest_rotations[hips_index].T
    heads, _tails = grounding.joint_positions(rig, globals_by_index, hips_index, translation)
    return heads


def hips_translation(rig: RigRest, take: SolvedTake) -> np.ndarray:
    return take.locations_by_bone[_HIPS] @ rig.rest_rotations[rig.index_of(_HIPS)].T


class GroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rig, self.bone_map = make_rig()
        self.rest = rest_landmarks(self.rig)

    def ankle_index(self, side: str) -> int:
        return self.rig.index_of(f"mixamorig:{side}Foot")

    def crouched(self, degrees: float) -> tuple[dict[str, np.ndarray], float]:
        frame = {name: points.copy() for name, points in self.rest.items()}
        pose = frame["pose"]
        ankle_moves = []
        for side in ("LEFT", "RIGHT"):
            hip, knee, ankle = (getattr(pose_landmarks, f"{side}_{name}") for name in ("HIP", "KNEE", "ANKLE"))
            thigh = (pose[knee] - pose[hip]) @ rotation_about((1.0, 0.0, 0.0), -degrees).T
            shin = (pose[ankle] - pose[knee]) @ rotation_about((1.0, 0.0, 0.0), degrees).T
            new_ankle = pose[hip] + thigh + shin
            feet = _FOOT_POINTS[side.title()]
            pose[feet] += new_ankle - pose[ankle]
            pose[knee] = pose[hip] + thigh
            ankle_moves.append(new_ankle - self.rest["pose"][ankle])
        shift = -np.mean(ankle_moves, axis=0)
        return {name: points + shift for name, points in frame.items()}, float(-shift[2])

    def test_crouch_lowers_the_hips_and_keeps_the_feet_on_the_floor(self) -> None:
        frames = [self.rest] * 10
        drop = 0.0
        for degrees in np.linspace(0.0, 40.0, 20):
            frame, drop = self.crouched(float(degrees))
            frames.append(frame)
        frames += [frames[-1]] * 10
        take = solve(make_cache(frames), self.rig, self.bone_map, MEDIAPIPE_TABLE, _NO_SMOOTHING)
        self.assertGreater(drop, 0.1)
        self.assertAlmostEqual(float(hips_translation(self.rig, take)[-1, 2]), -drop, delta=0.01)
        heads = solved_joint_positions(self.rig, take)
        for side in ("Left", "Right"):
            ankle = heads[:, self.ankle_index(side)]
            self.assertLess(float(np.abs(ankle - self.rig.heads[self.ankle_index(side)]).max()), 0.005, side)

    def test_planted_foot_is_locked_against_tracking_jitter(self) -> None:
        noise = np.random.default_rng(7).uniform(-0.005, 0.005, (40, 2))
        frames = []
        for frame_noise in noise:
            frame = {name: points.copy() for name, points in self.rest.items()}
            frame["pose"][_FOOT_POINTS["Left"], :2] += frame_noise
            frames.append(frame)
        cache = make_cache(frames)
        for root_motion, largest_slide in ((True, 0.001), (False, 0.001)):
            take = solve(cache, self.rig, self.bone_map, MEDIAPIPE_TABLE, SolveSettings(smoothing=0.0, root_motion=root_motion))
            ankle = solved_joint_positions(self.rig, take)[:, self.ankle_index("Left")]
            self.assertLess(float(np.ptp(ankle, axis=0).max()), largest_slide, f"root motion {root_motion}")

    def test_sway_over_planted_feet_moves_the_hips(self) -> None:
        times = np.arange(60) / 30.0
        sway = 0.04 * np.sin(2.0 * np.pi * 2.0 * times)
        foot_points = _FOOT_POINTS["Left"] + _FOOT_POINTS["Right"]
        frames = []
        for offset in sway:
            frame = {name: points + (offset, 0.0, 0.0) for name, points in self.rest.items()}
            frame["pose"][foot_points] = self.rest["pose"][foot_points]
            frames.append(frame)
        cache = make_cache(frames)
        with_travel = solve(cache, self.rig, self.bone_map, MEDIAPIPE_TABLE, SolveSettings(smoothing=0.0, root_motion=True))
        np.testing.assert_allclose(hips_translation(self.rig, with_travel)[:, 0], sway, atol=0.005)
        in_place = hips_translation(self.rig, solve(cache, self.rig, self.bone_map, MEDIAPIPE_TABLE, _NO_SMOOTHING))[:, 0]
        self.assertGreater(float(np.ptp(in_place)), 0.7 * float(np.ptp(sway)))
        self.assertLess(abs(float(in_place.mean())), 0.005)

    def test_still_actor_keeps_the_hips_at_rest(self) -> None:
        take = solve(make_cache([self.rest] * 10), self.rig, self.bone_map, MEDIAPIPE_TABLE, _NO_SMOOTHING)
        self.assertLess(float(np.abs(take.locations_by_bone[_HIPS]).max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
