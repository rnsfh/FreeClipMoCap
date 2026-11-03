import unittest

import numpy as np

from bl_ext.user_default.freeclipmocap.constants import hand_landmarks
from bl_ext.user_default.freeclipmocap.constants.bone_roles import Role
from bl_ext.user_default.freeclipmocap.solver.mediapipe_table import MEDIAPIPE_TABLE
from bl_ext.user_default.freeclipmocap.solver.pose_solver import SolveError, SolveSettings, SolvedTake, solve

from .synthetic_actor import make_cache, make_rig, rest_landmarks, rotation_about

_NO_SMOOTHING = SolveSettings(smoothing=0.0)
_TOLERANCE_DEGREES = 0.5


def rotation_degrees(quaternions: np.ndarray) -> np.ndarray:
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(quaternions[..., 0]), 0.0, 1.0)))


def bone(role_name: str) -> str:
    return f"mixamorig:{Role(role_name).value}"


def rotate_points(frame: dict[str, np.ndarray], keys: dict[str, list[int] | None], rotation: np.ndarray, pivot: np.ndarray) -> dict[str, np.ndarray]:
    rotated = {name: points.copy() for name, points in frame.items()}
    for name, indices in keys.items():
        selection = slice(None) if indices is None else indices
        rotated[name][selection] = (rotated[name][selection] - pivot) @ rotation.T + pivot
    return rotated


class PoseSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rig, self.bone_map = make_rig()
        self.rest = rest_landmarks(self.rig)

    def solve_frames(self, frames: list[dict[str, np.ndarray]], **cache_options: bool) -> SolvedTake:
        return solve(make_cache(frames, **cache_options), self.rig, self.bone_map, MEDIAPIPE_TABLE, _NO_SMOOTHING)

    def assert_still(self, take: SolvedTake, except_bones: set[str]) -> None:
        for bone_name, quaternions in take.rotations_by_bone.items():
            if bone_name not in except_bones:
                self.assertLess(rotation_degrees(quaternions).max(), _TOLERANCE_DEGREES, bone_name)

    def test_rest_pose_gives_no_rotation(self) -> None:
        take = self.solve_frames([self.rest, self.rest])
        self.assertIn(bone("LeftHandThumb3"), take.rotations_by_bone)
        self.assertIn(bone("Spine2"), take.rotations_by_bone)
        self.assertNotIn(bone("LeftToeBase"), take.rotations_by_bone)
        self.assert_still(take, set())
        self.assertLess(take.camera_tilt_degrees, 0.1)

    def test_elbow_bend_moves_only_the_forearm(self) -> None:
        elbow = self.rest["pose"][13]
        bend = rotation_about((0.0, 0.0, 1.0), -90.0)
        bent = rotate_points(self.rest, {"pose": [15, 17, 19, 21], "Left": None}, bend, elbow)
        take = self.solve_frames([bent])
        self.assertAlmostEqual(rotation_degrees(take.rotations_by_bone[bone("LeftForeArm")])[0], 90.0, delta=0.5)
        self.assert_still(take, {bone("LeftForeArm")})

    def test_body_turn_moves_only_the_hips(self) -> None:
        turn = rotation_about((0.0, 0.0, 1.0), 90.0)
        turned = rotate_points(self.rest, {"pose": None, "Left": None, "Right": None}, turn, np.zeros(3))
        take = self.solve_frames([turned])
        self.assertAlmostEqual(rotation_degrees(take.rotations_by_bone[bone("Hips")])[0], 90.0, delta=0.5)
        self.assert_still(take, {bone("Hips")})

    def test_tilted_camera_is_leveled(self) -> None:
        tilt = rotation_about((1.0, 0.0, 0.0), 20.0)
        tilted = rotate_points(self.rest, {"pose": None, "Left": None, "Right": None}, tilt, np.zeros(3))
        take = self.solve_frames([tilted])
        self.assertAlmostEqual(take.camera_tilt_degrees, 20.0, delta=0.1)
        self.assert_still(take, set())

    def test_shoulder_twist_turns_the_chest_through_the_spine(self) -> None:
        twist = rotation_about((0.0, 0.0, 1.0), 30.0)
        chest_center = (self.rest["pose"][11] + self.rest["pose"][12]) / 2
        upper_body = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        twisted = rotate_points(self.rest, {"pose": upper_body, "Left": None, "Right": None}, twist, chest_center)
        take = self.solve_frames([twisted])
        spine_degrees = [rotation_degrees(take.rotations_by_bone[bone(name)])[0] for name in ("Spine", "Spine1", "Spine2")]
        self.assertAlmostEqual(sum(spine_degrees), 30.0, delta=0.5)
        self.assertAlmostEqual(spine_degrees[0], 10.0, delta=0.5)
        self.assertLess(rotation_degrees(take.rotations_by_bone[bone("Hips")])[0], _TOLERANCE_DEGREES)

    def test_lost_hand_leaves_its_fingers_unanimated(self) -> None:
        take = self.solve_frames([self.rest], left_hand_found=False)
        self.assertNotIn(bone("LeftHandIndex1"), take.rotations_by_bone)
        self.assertIn(bone("RightHandIndex1"), take.rotations_by_bone)
        self.assertIn(bone("LeftHand"), take.rotations_by_bone)
        self.assertEqual(sum("left hand" in warning for warning in take.warnings), 1)

    def test_finger_bends_are_kept_within_human_limits(self) -> None:
        pivot = self.rest["Left"][hand_landmarks.INDEX_PIP]
        moved = [hand_landmarks.INDEX_DIP, hand_landmarks.INDEX_TIP]
        for axis, degrees, expected in (((0.0, 1.0, 0.0), 60.0, 60.0), ((0.0, 1.0, 0.0), -60.0, 5.0), ((0.0, 0.0, 1.0), 40.0, 5.0)):
            bent = rotate_points(self.rest, {"Left": moved}, rotation_about(axis, degrees), pivot)
            take = self.solve_frames([bent])
            angle = rotation_degrees(take.rotations_by_bone[bone("LeftHandIndex2")])[0]
            self.assertAlmostEqual(angle, expected, delta=0.5, msg=f"{degrees} degrees about {axis}")

    def test_legs_out_of_view_are_not_animated(self) -> None:
        visibility = np.ones((3, 33), dtype=np.float32)
        visibility[:, 25:33] = 0.02
        take = solve(make_cache([self.rest] * 3, pose_visibility=visibility), self.rig, self.bone_map, MEDIAPIPE_TABLE, _NO_SMOOTHING)
        for name in ("LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"):
            self.assertNotIn(bone(name), take.rotations_by_bone)
        self.assertIn(bone("Hips"), take.rotations_by_bone)
        self.assertTrue(any("LeftUpLeg" in warning for warning in take.warnings))
        self.assert_still(take, set())

    def test_points_seen_in_few_frames_are_not_used(self) -> None:
        visibility = np.ones((10, 33), dtype=np.float32)
        visibility[1:, 25:33] = 0.02
        take = solve(make_cache([self.rest] * 10, pose_visibility=visibility), self.rig, self.bone_map, MEDIAPIPE_TABLE, _NO_SMOOTHING)
        self.assertNotIn(bone("LeftUpLeg"), take.rotations_by_bone)

    def test_briefly_invisible_point_holds_its_last_position(self) -> None:
        bend = rotation_about((0.0, 0.0, 1.0), -90.0)
        bent = rotate_points(self.rest, {"pose": [15, 17, 19, 21], "Left": None}, bend, self.rest["pose"][13])
        visibility = np.ones((3, 33), dtype=np.float32)
        visibility[1, 15] = 0.1
        take = solve(make_cache([self.rest, bent, bent], pose_visibility=visibility), self.rig, self.bone_map, MEDIAPIPE_TABLE, _NO_SMOOTHING)
        forearm_degrees = rotation_degrees(take.rotations_by_bone[bone("LeftForeArm")])
        np.testing.assert_allclose(forearm_degrees, [0.0, 0.0, 90.0], atol=0.5)

    def test_single_left_right_mixup_is_skipped(self) -> None:
        mirrored = {name: points.copy() for name, points in self.rest.items()}
        for left_index, right_index in ((7, 8), (2, 5), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)):
            mirrored["pose"][[left_index, right_index]] = self.rest["pose"][[right_index, left_index]]
        frames = [self.rest] * 8 + [mirrored] + [self.rest] * 8
        take = self.solve_frames(frames)
        self.assert_still(take, set())
        self.assertTrue(any("1 frame" in warning for warning in take.warnings))

    def test_missing_body_is_an_error(self) -> None:
        cache = make_cache([self.rest])
        cache.pose.found[:] = False
        with self.assertRaises(SolveError):
            solve(cache, self.rig, self.bone_map, MEDIAPIPE_TABLE, _NO_SMOOTHING)

    def test_other_tracking_model_is_refused(self) -> None:
        with self.assertRaises(SolveError):
            solve(make_cache([self.rest], tracking_model="other"), self.rig, self.bone_map, MEDIAPIPE_TABLE, _NO_SMOOTHING)

    def test_resampling_keeps_the_duration(self) -> None:
        take = self.solve_frames([self.rest] * 31)
        resampled = take.resampled(24.0)
        self.assertEqual(resampled.frame_count, 25)
        self.assertAlmostEqual(resampled.times_seconds[-1], 1.0, places=6)
        self.assertEqual(resampled.rotations_by_bone[bone("Hips")].shape, (25, 4))


if __name__ == "__main__":
    unittest.main()
