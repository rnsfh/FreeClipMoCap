import unittest
from dataclasses import replace

import numpy as np

from bl_ext.user_default.freeclipmocap.capture.landmark_cache import LandmarkCache, VideoSource
from bl_ext.user_default.freeclipmocap.constants import pose_landmarks
from bl_ext.user_default.freeclipmocap.solver import rotations
from bl_ext.user_default.freeclipmocap.solver.leg_depth import with_legs_from_the_video
from bl_ext.user_default.freeclipmocap.solver.mediapipe_table import MEDIAPIPE_TABLE
from bl_ext.user_default.freeclipmocap.solver.pose_solver import SolveSettings, solve

from ..capture.calibration_fixtures import HEIGHT, WIDTH, looking_camera
from .synthetic_actor import make_cache, make_rig, rest_landmarks, rotation_about
from .test_grounding import solved_joint_positions

_CAMERA = looking_camera(np.array([0.3, -2.8, 1.3]), 1300.0)
_LEFT_LEG = (pose_landmarks.LEFT_HIP, pose_landmarks.LEFT_KNEE, pose_landmarks.LEFT_ANKLE)
_RIGHT_LEG = (pose_landmarks.RIGHT_HIP, pose_landmarks.RIGHT_KNEE, pose_landmarks.RIGHT_ANKLE)
_LEFT_SHIN_AND_FOOT = [
    pose_landmarks.LEFT_ANKLE,
    pose_landmarks.LEFT_HEEL,
    pose_landmarks.LEFT_FOOT_INDEX,
]


def filmed(frames: list[dict[str, np.ndarray]], fold_knees_metres: float) -> LandmarkCache:
    cache = make_cache(frames)
    image = np.empty((len(frames), pose_landmarks.LANDMARK_COUNT, 3), dtype=np.float32)
    world = np.empty_like(image)
    for frame_index, frame in enumerate(frames):
        camera_points = frame["pose"] @ _CAMERA.rotation.T + _CAMERA.translation
        pixels = camera_points[:, :2] / camera_points[:, 2:3] * _CAMERA.focal_length_pixels + (WIDTH / 2.0, HEIGHT / 2.0)
        image[frame_index, :, :2] = pixels / (WIDTH, HEIGHT)
        image[frame_index, :, 2] = 0.0
        hips_middle = (camera_points[pose_landmarks.LEFT_HIP] + camera_points[pose_landmarks.RIGHT_HIP]) / 2.0
        tracked = camera_points - hips_middle
        tracked[[pose_landmarks.LEFT_KNEE, pose_landmarks.RIGHT_KNEE], 2] += fold_knees_metres
        world[frame_index] = tracked
    return replace(
        cache,
        video=VideoSource(1, 1, 30.0, WIDTH, HEIGHT),
        pose=replace(cache.pose, image_landmarks=image, world_landmarks=world),
    )


def knee_bends(points: np.ndarray, leg: tuple[int, int, int]) -> np.ndarray:
    hip, knee, ankle = leg
    thigh = rotations.normalized(points[:, knee] - points[:, hip])
    shin = rotations.normalized(points[:, ankle] - points[:, knee])
    return np.degrees(np.arccos(np.clip(np.sum(thigh * shin, axis=-1), -1.0, 1.0)))


class LegDepthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rig, self.bone_map = make_rig()
        self.rest = rest_landmarks(self.rig)

    def walking(self) -> list[dict[str, np.ndarray]]:
        frames = []
        for angle in np.radians(np.linspace(0.0, 360.0, 40)):
            frame = {name: points.copy() for name, points in self.rest.items()}
            pose = frame["pose"]
            for leg, sign in ((_LEFT_LEG, 1.0), (_RIGHT_LEG, -1.0)):
                swing = rotation_about((1.0, 0.0, 0.0), sign * 25.0 * np.sin(angle))
                hip = pose[leg[0]].copy()
                lower = [leg[1], leg[2]] + ([pose_landmarks.LEFT_HEEL, pose_landmarks.LEFT_FOOT_INDEX] if sign > 0 else [pose_landmarks.RIGHT_HEEL, pose_landmarks.RIGHT_FOOT_INDEX])
                pose[lower] = (pose[lower] - hip) @ swing.T + hip
            frames.append(frame)
        return frames

    def test_knees_folded_in_depth_by_the_tracker_come_out_straight(self) -> None:
        cache = filmed(self.walking(), fold_knees_metres=0.12)
        self.assertGreater(float(np.median(knee_bends(cache.pose.world_landmarks, _LEFT_LEG))), 25.0)

        repaired = with_legs_from_the_video(cache, _CAMERA.focal_length_pixels).pose.world_landmarks.astype(np.float64)

        # These legs swing straight toward and away from the camera, the hardest case: where a bone is nearly side-on,
        # its two possible depths are close and the tracker's folded guess sometimes picks the wrong one.
        for leg in (_LEFT_LEG, _RIGHT_LEG):
            self.assertLess(float(np.median(knee_bends(repaired, leg))), 8.0)

    def test_a_knee_bent_in_the_picture_stays_bent(self) -> None:
        frames = self.walking()
        for frame in frames[:20]:
            pose = frame["pose"]
            knee = pose[pose_landmarks.LEFT_KNEE].copy()
            bend = rotation_about((0.0, 1.0, 0.0), 70.0)
            pose[_LEFT_SHIN_AND_FOOT] = (pose[_LEFT_SHIN_AND_FOOT] - knee) @ bend.T + knee
        cache = filmed(frames, fold_knees_metres=0.0)

        repaired = with_legs_from_the_video(cache, _CAMERA.focal_length_pixels).pose.world_landmarks.astype(np.float64)

        np.testing.assert_allclose(knee_bends(repaired, _LEFT_LEG)[:20], 70.0, atol=10.0)
        self.assertAlmostEqual(float(np.median(knee_bends(repaired, _LEFT_LEG)[:20])), 70.0, delta=5.0)

    def test_the_torso_and_arms_keep_the_tracker_s_points(self) -> None:
        cache = filmed(self.walking(), fold_knees_metres=0.12)
        repaired = with_legs_from_the_video(cache, _CAMERA.focal_length_pixels)
        upper_body = [pose_landmarks.NOSE, pose_landmarks.LEFT_SHOULDER, pose_landmarks.RIGHT_WRIST]
        tracked_middle = (cache.pose.world_landmarks[:, 23] + cache.pose.world_landmarks[:, 24]) / 2.0
        repaired_middle = (repaired.pose.world_landmarks[:, 23] + repaired.pose.world_landmarks[:, 24]) / 2.0
        np.testing.assert_allclose(
            repaired.pose.world_landmarks[:, upper_body] - repaired_middle[:, None],
            cache.pose.world_landmarks[:, upper_body] - tracked_middle[:, None],
            atol=0.02,
        )

    def test_a_hip_bend_is_kept_in_the_torso_not_turned_into_a_leg_slant(self) -> None:
        frame = {name: points.copy() for name, points in self.rest.items()}
        pose = frame["pose"]
        hips_middle = (pose[pose_landmarks.LEFT_HIP] + pose[pose_landmarks.RIGHT_HIP]) / 2.0
        upper_body = [index for index in range(pose_landmarks.LANDMARK_COUNT) if index <= pose_landmarks.RIGHT_THUMB]
        lean = rotation_about((1.0, 0.0, 0.0), 12.0)
        pose[upper_body] = (pose[upper_body] - hips_middle) @ lean.T + hips_middle
        for side in ("Left", "Right"):
            frame[side] = (frame[side] - hips_middle) @ lean.T + hips_middle

        take = solve(make_cache([frame] * 10), self.rig, self.bone_map, MEDIAPIPE_TABLE, SolveSettings(smoothing=0.0))

        heads = solved_joint_positions(self.rig, take)
        up = np.array([0.0, 0.0, 1.0])
        ankles = (heads[:, self.rig.index_of("mixamorig:LeftFoot")] + heads[:, self.rig.index_of("mixamorig:RightFoot")]) / 2.0
        hips = heads[:, self.rig.index_of("mixamorig:Hips")]
        neck = heads[:, self.rig.index_of("mixamorig:Neck")]
        legs_degrees = np.degrees(np.arccos(rotations.normalized(hips - ankles) @ up))
        torso_degrees = np.degrees(np.arccos(rotations.normalized(neck - hips) @ up))
        self.assertLess(float(legs_degrees.max()), 6.0)
        self.assertGreater(float(torso_degrees.min()), 6.0)


if __name__ == "__main__":
    unittest.main()
