import unittest
from dataclasses import replace

import numpy as np

from bl_ext.user_default.freeclipmocap.capture.camera_setup import CalibratedCamera
from bl_ext.user_default.freeclipmocap.capture.landmark_cache import HandTrack, LandmarkCache
from bl_ext.user_default.freeclipmocap.constants import hand_landmarks, pose_landmarks
from bl_ext.user_default.freeclipmocap.solver import rotations
from bl_ext.user_default.freeclipmocap.solver.mediapipe_table import MEDIAPIPE_TABLE
from bl_ext.user_default.freeclipmocap.solver.pose_solver import SolveSettings, solve
from bl_ext.user_default.freeclipmocap.solver.triangulation import TriangulationError, triangulated_cache

from ..capture import body_fixtures
from ..capture.calibration_fixtures import FRAME_RATE, HEIGHT, WIDTH, SyntheticCamera
from .synthetic_actor import make_rig

_BODY_POINTS = [
    getattr(pose_landmarks, f"{side}_{name}")
    for side in ("LEFT", "RIGHT")
    for name in ("SHOULDER", "ELBOW", "WRIST", "HIP", "KNEE", "ANKLE", "HEEL", "FOOT_INDEX")
]
_TRACKER_TO_WORLD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_HAND_SHAPE = np.array(
    [[0.0, 0.0, 0.0]]
    + [[0.02 * finger - 0.04, 0.0, -0.03 - 0.025 * joint] for finger in range(5) for joint in range(4)]
)


def calibrated(camera: SyntheticCamera, file_name: str) -> CalibratedCamera:
    return CalibratedCamera(
        video_file_name=file_name,
        image_width=WIDTH,
        image_height=HEIGHT,
        camera_matrix=camera.camera_matrix,
        distortion=np.zeros(5),
        rotation=camera.rotation,
        translation=camera.translation,
        lens_error_pixels=0.0,
        picture_delay_correction_seconds=0.0,
    )


def hand_in_world(time_seconds: float) -> np.ndarray:
    wrist = body_fixtures.actor_points(time_seconds)[pose_landmarks.LEFT_WRIST]
    turn = rotations.rotation_between(np.array([0.0, 0.0, -1.0]), np.array([0.3, -0.2, -1.0]))
    return wrist + _HAND_SHAPE @ turn.T


def with_left_hand(cache: LandmarkCache, camera: SyntheticCamera, started_at_seconds: float, seen_frames: int) -> LandmarkCache:
    frame_count = cache.frame_count
    image = np.full((frame_count, hand_landmarks.LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    world = np.full_like(image, np.nan)
    for frame_index in range(min(seen_frames, frame_count)):
        points = hand_in_world(started_at_seconds + frame_index / FRAME_RATE)
        camera_points = points @ camera.rotation.T + camera.translation
        pixels = camera_points[:, :2] / camera_points[:, 2:3] * camera.focal_length_pixels + (WIDTH / 2.0, HEIGHT / 2.0)
        image[frame_index, :, 0] = pixels[:, 0] / WIDTH
        image[frame_index, :, 1] = pixels[:, 1] / HEIGHT
        world[frame_index] = (points - points.mean(axis=0)) @ camera.rotation.T
    found = np.zeros(frame_count, dtype=bool)
    found[: min(seen_frames, frame_count)] = True
    track = HandTrack(found=found, image_landmarks=image, world_landmarks=world, score=found.astype(np.float32))
    return replace(cache, left_hand=track)


class TriangulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.front, cls.side = body_fixtures.filmed_take()
        cls.cameras = (
            calibrated(body_fixtures.FRONT_CAMERA, "front.mp4"),
            calibrated(body_fixtures.SIDE_CAMERA, "side.mp4"),
        )
        cls.delays = (0.0, body_fixtures.SIDE_STARTED_LATER_SECONDS)
        cls.cache, cls.report = triangulated_cache((cls.front, cls.side), cls.cameras, cls.delays)

    def test_two_phones_place_the_actor_in_the_world(self) -> None:
        frames = np.arange(0, self.cache.frame_count - int(FRAME_RATE), 5)
        truth = np.stack([body_fixtures.actor_points(frame / FRAME_RATE)[_BODY_POINTS] for frame in frames])
        found = self.cache.pose.world_landmarks[frames][:, _BODY_POINTS].astype(np.float64) @ _TRACKER_TO_WORLD
        errors = np.linalg.norm(found - truth, axis=-1)
        # Each phone places each joint up to ~1.5 cm differently (as MediaPipe does), so a few cm is the floor.
        self.assertLess(float(np.nanmedian(errors)), 0.035)
        self.assertLess(float(np.nanpercentile(errors, 95)), 0.07)
        self.assertGreater(self.report.body_share, 0.95)
        self.assertLess(self.report.median_error_pixels, 6.0)

    def test_frames_where_one_phone_swapped_left_and_right_are_undone(self) -> None:
        swapped_share = self.report.swapped_frame_counts[0] / self.side.frame_count
        self.assertAlmostEqual(swapped_share, body_fixtures.SWAPPED_SHARE, delta=0.015)

    def test_the_take_keeps_camera_1_s_frames_and_pictures(self) -> None:
        np.testing.assert_array_equal(self.cache.timestamps_milliseconds, self.front.timestamps_milliseconds)
        np.testing.assert_array_equal(self.cache.pose.image_landmarks, self.front.pose.image_landmarks)

    def test_a_hand_seen_by_both_phones_or_by_one_keeps_its_shape(self) -> None:
        front = with_left_hand(self.front, body_fixtures.FRONT_CAMERA, 0.0, 120)
        side = with_left_hand(self.side, body_fixtures.SIDE_CAMERA, body_fixtures.SIDE_STARTED_LATER_SECONDS, 40)
        cache, report = triangulated_cache((front, side), self.cameras, self.delays)

        self.assertGreater(report.left_hand.triangulated_share, 0.0)
        self.assertGreater(report.left_hand.one_camera_share, 0.0)
        self.assertFalse(cache.left_hand.found[130:].any())
        for frame in (10, 100):
            truth = hand_in_world(frame / FRAME_RATE)
            found = cache.left_hand.world_landmarks[frame].astype(np.float64) @ _TRACKER_TO_WORLD
            np.testing.assert_allclose(found - found[0], truth - truth[0], atol=0.01)

    def test_the_solver_keeps_the_triangulated_knees(self) -> None:
        rig, bone_map = make_rig()
        take = solve(self.cache, rig, bone_map, MEDIAPIPE_TABLE, SolveSettings(smoothing=0.5, level_by_body=False))
        self.assertEqual(take.camera_tilt_degrees, 0.0)
        points = self.cache.pose.world_landmarks.astype(np.float64)
        for side in ("Left", "Right"):
            hip, knee, ankle = (getattr(pose_landmarks, f"{side.upper()}_{name}") for name in ("HIP", "KNEE", "ANKLE"))
            thigh = rotations.normalized(points[:, knee] - points[:, hip])
            shin = rotations.normalized(points[:, ankle] - points[:, knee])
            tracked_degrees = np.degrees(np.arccos(np.clip(np.sum(thigh * shin, axis=-1), -1.0, 1.0)))
            solved = rotations.matrices_from_quaternions(take.rotations_by_bone[f"mixamorig:{side}Leg"])
            solved_degrees = np.degrees(np.arccos(np.clip((np.trace(solved, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)))
            # The foot lock may turn a knee at most 3° from the tracked bend.
            self.assertLess(float(np.median(solved_degrees)), float(np.nanmedian(tracked_degrees)) + 3.5, side)
            self.assertLess(float(np.nanmedian(tracked_degrees)), 10.0, side)

    def test_one_camera_is_refused(self) -> None:
        with self.assertRaises(TriangulationError):
            triangulated_cache((self.front,), self.cameras[:1], (0.0,))


if __name__ == "__main__":
    unittest.main()
