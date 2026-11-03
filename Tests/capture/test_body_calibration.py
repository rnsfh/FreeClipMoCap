import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from bl_ext.user_default.freeclipmocap.capture.body_calibration import calibrate_cameras_from_bodies
from bl_ext.user_default.freeclipmocap.capture.camera_calibration import CalibrationError
from bl_ext.user_default.freeclipmocap.capture.camera_setup import (
    BodyCalibrationInputs,
    CalibrationInputs,
    CalibrationVideo,
    CameraSetup,
    FloorSource,
    TypedLens,
    find_current_setup,
)
from bl_ext.user_default.freeclipmocap.capture.landmark_cache import LandmarkCache
from bl_ext.user_default.freeclipmocap.constants.camera_lens import FULL_FRAME_WIDTH_MILLIMETRES

from . import body_fixtures as fixtures
from .calibration_fixtures import FRAME_RATE, SyntheticCamera, looking_camera

_VIDEO_NAMES = ["front.mp4", "side.mp4", "back.mp4"]
_LARGEST_POSITION_ERROR_METRES = 0.04
_LARGEST_ROTATION_ERROR_DEGREES = 1.0
_THIRD_CAMERA = looking_camera(np.array([-2.2, -2.1, 1.4]), 1150.0)
_THIRD_STARTED_LATER_SECONDS = 0.8


def rotation_difference_degrees(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(first @ second.T) - 1.0) / 2.0, -1.0, 1.0))))


def typed_lens_of(camera: SyntheticCamera) -> TypedLens:
    return TypedLens(camera.focal_length_pixels * FULL_FRAME_WIDTH_MILLIMETRES / fixtures.WIDTH, True)


def body_inputs(cameras: tuple[SyntheticCamera, ...], start_delays_seconds: tuple[float, ...]) -> CalibrationInputs:
    return CalibrationInputs(
        videos=tuple(CalibrationVideo(name, 1, 1) for name in _VIDEO_NAMES[: len(cameras)]),
        start_delays_seconds=start_delays_seconds,
        saved_lenses=(None,) * len(cameras),
        typed_lenses=tuple(typed_lens_of(camera) for camera in cameras),
        square_size_metres=0.048,
        edge_below_squares_metres=0.0285,
        first_frame=0,
        stop_frame=0,
        body=BodyCalibrationInputs(
            float(np.linalg.norm(fixtures.SIDE_CAMERA.position - fixtures.FRONT_CAMERA.position)), fixtures.CAPTURE_SETTINGS
        ),
    )


class BodyCalibrationTests(unittest.TestCase):
    front: LandmarkCache
    side: LandmarkCache

    @classmethod
    def setUpClass(cls) -> None:
        cls.front, cls.side = fixtures.filmed_take()

    def two_camera_inputs(self, sound_error_seconds: float = 0.0) -> CalibrationInputs:
        return body_inputs(
            (fixtures.FRONT_CAMERA, fixtures.SIDE_CAMERA), (0.0, fixtures.SIDE_STARTED_LATER_SECONDS + sound_error_seconds)
        )

    def assert_matches_the_filmed_cameras(
        self, setup: CameraSetup, filmed_cameras: tuple[SyntheticCamera, ...], largest_position_error_metres: float
    ) -> None:
        for calibrated, filmed in zip(setup.cameras, filmed_cameras):
            self.assertLess(np.linalg.norm(calibrated.position - filmed.position), largest_position_error_metres)
            self.assertLess(rotation_difference_degrees(calibrated.rotation, filmed.rotation), _LARGEST_ROTATION_ERROR_DEGREES)

    def test_finds_both_cameras_and_the_floor_from_the_walking_actor(self) -> None:
        setup = calibrate_cameras_from_bodies([self.front, self.side], _VIDEO_NAMES[:2], self.two_camera_inputs())

        self.assert_matches_the_filmed_cameras(setup, (fixtures.FRONT_CAMERA, fixtures.SIDE_CAMERA), _LARGEST_POSITION_ERROR_METRES)
        self.assertEqual(setup.floor_source, FloorSource.FEET)
        self.assertIsNone(setup.floor_video_frames)
        distance_between_cameras = np.linalg.norm(setup.cameras[1].position - setup.cameras[0].position)
        self.assertAlmostEqual(distance_between_cameras, setup.inputs.body.cameras_apart_metres, places=6)
        self.assertLess(max(camera.lens_error_pixels for camera in setup.cameras), 4.0)

    def test_corrects_a_sync_offset_that_is_slightly_off(self) -> None:
        sound_error_seconds = 0.03

        setup = calibrate_cameras_from_bodies([self.front, self.side], _VIDEO_NAMES[:2], self.two_camera_inputs(sound_error_seconds))

        self.assertAlmostEqual(setup.cameras[1].picture_delay_correction_seconds, -sound_error_seconds, delta=0.008)
        self.assert_matches_the_filmed_cameras(setup, (fixtures.FRONT_CAMERA, fixtures.SIDE_CAMERA), _LARGEST_POSITION_ERROR_METRES)

    def test_a_third_camera_is_placed_from_the_first_two(self) -> None:
        frame_count = int((fixtures.TAKE_SECONDS - _THIRD_STARTED_LATER_SECONDS) * FRAME_RATE)
        third = fixtures.filmed_cache(_THIRD_CAMERA, _THIRD_STARTED_LATER_SECONDS, frame_count, seed=13)
        cameras = (fixtures.FRONT_CAMERA, fixtures.SIDE_CAMERA, _THIRD_CAMERA)
        inputs = body_inputs(cameras, (0.0, fixtures.SIDE_STARTED_LATER_SECONDS, _THIRD_STARTED_LATER_SECONDS))

        setup = calibrate_cameras_from_bodies([self.front, self.side, third], _VIDEO_NAMES, inputs)

        self.assert_matches_the_filmed_cameras(setup, cameras, 2.0 * _LARGEST_POSITION_ERROR_METRES)

    def test_a_camera_without_a_typed_or_saved_lens_is_refused(self) -> None:
        inputs = replace(self.two_camera_inputs(), typed_lenses=(typed_lens_of(fixtures.FRONT_CAMERA), None))

        with self.assertRaisesRegex(CalibrationError, "side.mp4: calibrating from the body needs the lens"):
            calibrate_cameras_from_bodies([self.front, self.side], _VIDEO_NAMES[:2], inputs)

    def test_cameras_that_never_see_the_body_together_are_refused(self) -> None:
        unseen = replace(self.side, pose=replace(self.side.pose, visibility=np.zeros_like(self.side.pose.visibility)))

        with self.assertRaisesRegex(CalibrationError, "too few frames"):
            calibrate_cameras_from_bodies([self.front, unseen], _VIDEO_NAMES[:2], self.two_camera_inputs())

    def test_the_saved_setup_records_the_body_inputs(self) -> None:
        setup = calibrate_cameras_from_bodies([self.front, self.side], _VIDEO_NAMES[:2], self.two_camera_inputs())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "front.mp4.calibration.json"
            setup.save(path)

            loaded = CameraSetup.load(path)
            moved_phone = replace(setup.inputs, body=replace(setup.inputs.body, cameras_apart_metres=3.0))

            self.assertEqual(loaded.inputs, setup.inputs)
            self.assertEqual(loaded.floor_source, FloorSource.FEET)
            self.assertIsNone(find_current_setup(path, moved_phone))
            self.assertIsNone(find_current_setup(path, replace(setup.inputs, body=None)))


if __name__ == "__main__":
    unittest.main()
