import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from bl_ext.user_default.freeclipmocap.capture.board_detection import detect_board, save_board_image
from bl_ext.user_default.freeclipmocap.capture.board_sightings import BoardSightings
from bl_ext.user_default.freeclipmocap.capture.camera_calibration import CalibrationError, calibrate_cameras
from bl_ext.user_default.freeclipmocap.capture.camera_setup import (
    CalibratedCamera,
    CameraSetup,
    FloorSource,
    calibration_path_for,
)
from bl_ext.user_default.freeclipmocap.capture.capture_progress import CaptureProgress
from bl_ext.user_default.freeclipmocap.constants import calibration_board

from . import calibration_fixtures as fixtures

_SIDE_STARTED_LATER_SECONDS = 0.37
_LARGEST_POSITION_ERROR_METRES = 0.03
_LARGEST_ROTATION_ERROR_DEGREES = 0.7
_LARGEST_FOCAL_LENGTH_ERROR_SHARE = 0.02


def rotation_difference_degrees(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(first @ second.T) - 1.0) / 2.0, -1.0, 1.0))))


def frames_before(sightings: BoardSightings, stop_seconds: float) -> BoardSightings:
    return frames_kept(sightings, sightings.timestamps_seconds < stop_seconds)


def frames_outside(sightings: BoardSightings, start_seconds: float, stop_seconds: float) -> BoardSightings:
    times = sightings.timestamps_seconds
    return frames_kept(sightings, (times < start_seconds) | (times >= stop_seconds))


def frames_kept(sightings: BoardSightings, kept: np.ndarray) -> BoardSightings:
    return replace(
        sightings,
        frame_indices=sightings.frame_indices[kept],
        timestamps_seconds=sightings.timestamps_seconds[kept],
        corner_points=sightings.corner_points[kept],
    )


class CameraCalibrationTests(unittest.TestCase):
    temporary_folder: tempfile.TemporaryDirectory
    front_sightings: BoardSightings
    side_sightings: BoardSightings

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_folder = tempfile.TemporaryDirectory()
        folder = Path(cls.temporary_folder.name)
        front_video = fixtures.write_board_video(
            folder / "front.mp4",
            fixtures.FRONT_CAMERA,
            0.0,
            int(fixtures.CALIBRATION_SECONDS * fixtures.FRAME_RATE),
            fixtures.calibration_board_pose,
        )
        side_video = fixtures.write_board_video(
            folder / "side.mp4",
            fixtures.SIDE_CAMERA,
            _SIDE_STARTED_LATER_SECONDS,
            int((fixtures.CALIBRATION_SECONDS - _SIDE_STARTED_LATER_SECONDS) * fixtures.FRAME_RATE),
            fixtures.calibration_board_pose,
        )
        cls.front_sightings = detect_board(front_video, 0, 0, CaptureProgress(), threading.Event())
        cls.side_sightings = detect_board(side_video, 0, 0, CaptureProgress(), threading.Event())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_folder.cleanup()

    def sightings_before(self, stop_seconds: float) -> list[BoardSightings]:
        return [
            frames_before(self.front_sightings, stop_seconds),
            frames_before(self.side_sightings, stop_seconds - _SIDE_STARTED_LATER_SECONDS),
        ]

    def sightings_outside(self, start_seconds: float, stop_seconds: float) -> list[BoardSightings]:
        return [
            frames_outside(self.front_sightings, start_seconds, stop_seconds),
            frames_outside(
                self.side_sightings, start_seconds - _SIDE_STARTED_LATER_SECONDS, stop_seconds - _SIDE_STARTED_LATER_SECONDS
            ),
        ]

    def calibrate(self, sightings: list[BoardSightings], side_started_later_seconds: float = _SIDE_STARTED_LATER_SECONDS) -> CameraSetup:
        return calibrate_cameras(
            sightings, [0.0, side_started_later_seconds], fixtures.SQUARE_SIZE_METRES, fixtures.PAPER_HEIGHT_METRES
        )

    def assert_matches_the_filmed_cameras(self, setup: CameraSetup) -> None:
        for calibrated, filmed in zip(setup.cameras, (fixtures.FRONT_CAMERA, fixtures.SIDE_CAMERA)):
            self.assertLess(np.linalg.norm(calibrated.position - filmed.position), _LARGEST_POSITION_ERROR_METRES)
            self.assertLess(rotation_difference_degrees(calibrated.rotation, filmed.rotation), _LARGEST_ROTATION_ERROR_DEGREES)
            self.assertAlmostEqual(
                calibrated.camera_matrix[0, 0] / filmed.focal_length_pixels, 1.0, delta=_LARGEST_FOCAL_LENGTH_ERROR_SHARE
            )

    def test_finds_both_cameras_and_the_floor_from_the_board_standing_on_it(self) -> None:
        setup = self.calibrate(self.sightings_before(fixtures.FLAT_FROM_SECONDS))

        self.assert_matches_the_filmed_cameras(setup)
        self.assertEqual(setup.floor_source, FloorSource.BOARD_UPRIGHT)
        camera_started_at_seconds = (0.0, _SIDE_STARTED_LATER_SECONDS)[setup.floor_camera_index]
        floor_shot_frame = (fixtures.UPRIGHT_FROM_SECONDS - camera_started_at_seconds) * fixtures.FRAME_RATE
        self.assertGreaterEqual(setup.floor_video_frames[0], floor_shot_frame - 1)
        self.assertLess(setup.pair_error_pixels[0], 1.0)

    def test_a_board_lying_flat_also_gives_the_floor(self) -> None:
        setup = self.calibrate(self.sightings_outside(fixtures.UPRIGHT_FROM_SECONDS, fixtures.FLAT_FROM_SECONDS))

        self.assert_matches_the_filmed_cameras(setup)
        self.assertEqual(setup.floor_source, FloorSource.BOARD_FLAT)

    def test_corrects_a_sync_offset_that_is_slightly_off(self) -> None:
        sound_error_seconds = 0.025

        setup = self.calibrate([self.front_sightings, self.side_sightings], _SIDE_STARTED_LATER_SECONDS + sound_error_seconds)

        self.assertAlmostEqual(setup.cameras[1].picture_delay_correction_seconds, -sound_error_seconds, delta=0.008)
        self.assert_matches_the_filmed_cameras(setup)

    def test_without_a_floor_shot_the_floor_is_left_to_the_feet(self) -> None:
        setup = self.calibrate(self.sightings_before(fixtures.UPRIGHT_FROM_SECONDS))

        self.assertFalse(setup.floor_found)
        self.assertEqual(setup.floor_source, FloorSource.NONE)
        self.assertIsNone(setup.floor_video_frames)
        distance_between_cameras = np.linalg.norm(setup.cameras[1].position - setup.cameras[0].position)
        filmed_distance = np.linalg.norm(fixtures.SIDE_CAMERA.position - fixtures.FRONT_CAMERA.position)
        self.assertAlmostEqual(distance_between_cameras, filmed_distance, delta=_LARGEST_POSITION_ERROR_METRES)

    def test_cameras_that_never_see_the_board_together_are_refused(self) -> None:
        front_close_up_only = frames_before(self.front_sightings, fixtures.CLOSE_UP_SECONDS)
        side_close_up_only = frames_before(self.side_sightings, 2.0 * fixtures.CLOSE_UP_SECONDS - _SIDE_STARTED_LATER_SECONDS)
        with self.assertRaises(CalibrationError):
            self.calibrate([front_close_up_only, side_close_up_only])

    def test_saved_setup_loads_with_the_same_values_and_projects_like_the_filmed_camera(self) -> None:
        setup = self.calibrate([self.front_sightings, self.side_sightings])
        path = calibration_path_for(Path(self.temporary_folder.name) / "front.mp4")
        setup.save(path)

        loaded = CameraSetup.load(path)

        self.assertEqual(path.name, "front.mp4.calibration.json")
        self.assertEqual(loaded.floor_source, setup.floor_source)
        self.assertEqual(loaded.floor_video_frames, setup.floor_video_frames)
        self.assertEqual([camera.video_file_name for camera in loaded.cameras], ["front.mp4", "side.mp4"])
        np.testing.assert_allclose(loaded.cameras[1].rotation, setup.cameras[1].rotation)
        self.assertFalse(loaded.cameras[0].camera_matrix.flags.writeable)
        points = np.array([[0.0, 0.0, 1.0], [0.3, 0.2, 0.2], [-0.2, -0.1, 1.5]])
        for calibrated, filmed in zip(loaded.cameras, (fixtures.FRONT_CAMERA, fixtures.SIDE_CAMERA)):
            projected = (filmed.camera_matrix @ (points @ filmed.rotation.T + filmed.translation).T).T
            np.testing.assert_allclose(calibrated.project(points), projected[:, :2] / projected[:, 2:], atol=25.0)

    def test_projection_matches_opencv_with_distortion(self) -> None:
        camera = CalibratedCamera(
            video_file_name="clip.mp4",
            image_width=1920,
            image_height=1080,
            camera_matrix=np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]]),
            distortion=np.array([0.08, -0.2, 0.001, -0.002, 0.05]),
            rotation=cv2.Rodrigues(np.array([0.2, -0.4, 0.1]))[0],
            translation=np.array([0.1, -0.3, 2.5]),
            lens_error_pixels=0.2,
            picture_delay_correction_seconds=0.0,
        )
        points = np.random.default_rng(6).uniform(-0.8, 0.8, size=(20, 3))

        expected, _jacobian = cv2.projectPoints(
            points, cv2.Rodrigues(camera.rotation)[0], camera.translation, camera.camera_matrix, camera.distortion
        )

        np.testing.assert_allclose(camera.project(points), expected.reshape(-1, 2), atol=1e-6)


class BoardTests(unittest.TestCase):
    def test_printed_board_image_is_found_with_every_corner(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "board.png"
            save_board_image(path)
            page = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            video_path = Path(folder) / "board.mp4"
            small_page = cv2.resize(page, (page.shape[1] // 4, page.shape[0] // 4), interpolation=cv2.INTER_AREA)
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, small_page.shape[::-1])
            writer.write(cv2.cvtColor(small_page, cv2.COLOR_GRAY2BGR))
            writer.release()

            sightings = detect_board(video_path, 0, 0, CaptureProgress(), threading.Event())

        self.assertEqual(page.shape, (3508, 2480))
        self.assertEqual(int(sightings.corners_seen[0]), calibration_board.INNER_CORNER_COUNT)

    def test_corners_are_interpolated_only_between_consecutive_frames(self) -> None:
        corner_points = np.full((3, calibration_board.INNER_CORNER_COUNT, 2), np.nan, dtype=np.float32)
        corner_points[0, 0] = (100.0, 200.0)
        corner_points[1, 0] = (110.0, 220.0)
        corner_points[2, 0] = (500.0, 500.0)
        sightings = BoardSightings(
            video_file_name="clip.mp4",
            image_width=1920,
            image_height=1080,
            frame_rate=30.0,
            frame_indices=np.array([0, 1, 5], dtype=np.int32),
            timestamps_seconds=np.array([0.0, 1.0 / 30.0, 5.0 / 30.0]),
            corner_points=corner_points,
        )

        points = sightings.interpolated_at(np.array([0.25 / 30.0, 3.0 / 30.0, -1.0]))

        np.testing.assert_allclose(points[0, 0], (102.5, 205.0), atol=1e-4)
        self.assertTrue(np.isnan(points[0, 1, 0]))
        self.assertTrue(np.isnan(points[1, 0, 0]))
        self.assertTrue(np.isnan(points[2, 0, 0]))


if __name__ == "__main__":
    unittest.main()
