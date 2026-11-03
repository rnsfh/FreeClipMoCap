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
    CalibrationInputs,
    CalibrationVideo,
    CameraSetup,
    FloorSource,
    TypedLens,
    calibration_path_for,
    find_current_setup,
)
from bl_ext.user_default.freeclipmocap.capture.capture_progress import CaptureProgress
from bl_ext.user_default.freeclipmocap.capture.saved_lens import (
    SavedLens,
    SavedLensError,
    delete_lens,
    lens_path,
    load_lenses,
)
from bl_ext.user_default.freeclipmocap.constants import calibration_board
from bl_ext.user_default.freeclipmocap.constants.camera_lens import FULL_FRAME_WIDTH_MILLIMETRES

from . import calibration_fixtures as fixtures

_SIDE_STARTED_LATER_SECONDS = fixtures.SIDE_STARTED_LATER_SECONDS
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


def calibration_inputs(
    video_paths: list[Path],
    side_started_later_seconds: float = _SIDE_STARTED_LATER_SECONDS,
    saved_lenses: tuple[SavedLens | None, ...] = (None, None),
    typed_lenses: tuple[TypedLens | None, ...] = (None, None),
) -> CalibrationInputs:
    return CalibrationInputs(
        videos=tuple(CalibrationVideo.of(path) for path in video_paths),
        start_delays_seconds=(0.0, side_started_later_seconds),
        saved_lenses=saved_lenses,
        typed_lenses=typed_lenses,
        square_size_metres=fixtures.SQUARE_SIZE_METRES,
        edge_below_squares_metres=fixtures.EDGE_BELOW_SQUARES_METRES,
        first_frame=0,
        stop_frame=0,
    )


def typed_lens_of(filmed: fixtures.SyntheticCamera, share_off: float, keep_fixed: bool) -> TypedLens:
    focal_length_millimetres = filmed.focal_length_pixels * (1.0 + share_off) * FULL_FRAME_WIDTH_MILLIMETRES / fixtures.WIDTH
    return TypedLens(focal_length_millimetres, keep_fixed)


def saved_lens_of(camera: CalibratedCamera, name: str) -> SavedLens:
    return SavedLens.from_arrays(
        name, camera.image_width, camera.image_height, camera.camera_matrix, camera.distortion, camera.lens_error_pixels
    )


class CameraCalibrationTests(unittest.TestCase):
    temporary_folder: tempfile.TemporaryDirectory
    video_paths: list[Path]
    front_sightings: BoardSightings
    side_sightings: BoardSightings

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_folder = tempfile.TemporaryDirectory()
        cls.video_paths = list(fixtures.shared_calibration_videos())
        cls.front_sightings = detect_board(cls.video_paths[0], 0, 0, CaptureProgress(), threading.Event())
        cls.side_sightings = detect_board(cls.video_paths[1], 0, 0, CaptureProgress(), threading.Event())

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

    def calibrate(
        self,
        sightings: list[BoardSightings],
        side_started_later_seconds: float = _SIDE_STARTED_LATER_SECONDS,
        saved_lenses: tuple[SavedLens | None, ...] = (None, None),
        typed_lenses: tuple[TypedLens | None, ...] = (None, None),
    ) -> CameraSetup:
        inputs = calibration_inputs(self.video_paths, side_started_later_seconds, saved_lenses, typed_lenses)
        return calibrate_cameras(sightings, inputs)

    def saved_lenses_from_a_full_calibration(self) -> tuple[SavedLens, SavedLens]:
        setup = self.calibrate([self.front_sightings, self.side_sightings])
        return saved_lens_of(setup.cameras[0], "Front"), saved_lens_of(setup.cameras[1], "Side")

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

    def test_the_floor_follows_the_distance_below_the_squares(self) -> None:
        sightings = self.sightings_before(fixtures.FLAT_FROM_SECONDS)
        larger_distance_metres = 0.03
        inputs = calibration_inputs(self.video_paths)

        on_paper = calibrate_cameras(sightings, inputs)
        on_a_taller_stand = calibrate_cameras(
            sightings, replace(inputs, edge_below_squares_metres=inputs.edge_below_squares_metres + larger_distance_metres)
        )

        self.assertLess(abs(on_paper.cameras[0].position[2] - fixtures.FRONT_CAMERA.position[2]), _LARGEST_POSITION_ERROR_METRES)
        for paper_camera, stand_camera in zip(on_paper.cameras, on_a_taller_stand.cameras):
            self.assertAlmostEqual(stand_camera.position[2] - paper_camera.position[2], larger_distance_metres, delta=0.002)

    def test_saved_lenses_stay_fixed_and_no_close_ups_are_needed(self) -> None:
        saved_lenses = self.saved_lenses_from_a_full_calibration()
        without_close_ups = self.sightings_outside(0.0, 2.0 * fixtures.CLOSE_UP_SECONDS)

        setup = self.calibrate(without_close_ups, saved_lenses=saved_lenses)

        self.assert_matches_the_filmed_cameras(setup)
        for camera, saved_lens in zip(setup.cameras, saved_lenses):
            np.testing.assert_array_equal(camera.camera_matrix, saved_lens.camera_matrix)
            np.testing.assert_array_equal(camera.distortion, saved_lens.distortion)

    def test_a_saved_lens_and_a_measured_one_work_together(self) -> None:
        front_lens, _side_lens = self.saved_lenses_from_a_full_calibration()

        setup = self.calibrate([self.front_sightings, self.side_sightings], saved_lenses=(front_lens, None))

        self.assert_matches_the_filmed_cameras(setup)
        np.testing.assert_array_equal(setup.cameras[0].camera_matrix, front_lens.camera_matrix)

    def test_typed_focal_lengths_kept_fixed_need_no_close_ups(self) -> None:
        typed_lenses = (typed_lens_of(fixtures.FRONT_CAMERA, 0.0, True), typed_lens_of(fixtures.SIDE_CAMERA, 0.0, True))

        setup = self.calibrate(self.sightings_outside(0.0, 2.0 * fixtures.CLOSE_UP_SECONDS), typed_lenses=typed_lenses)

        self.assert_matches_the_filmed_cameras(setup)
        for camera, typed_lens in zip(setup.cameras, typed_lenses):
            self.assertAlmostEqual(camera.focal_length_millimetres, typed_lens.focal_length_millimetres, places=9)
            np.testing.assert_array_equal(camera.distortion, np.zeros(5))

    def test_typed_focal_lengths_that_are_off_are_corrected_by_the_board(self) -> None:
        typed_lenses = (typed_lens_of(fixtures.FRONT_CAMERA, 0.1, False), typed_lens_of(fixtures.SIDE_CAMERA, -0.1, False))

        setup = self.calibrate([self.front_sightings, self.side_sightings], typed_lenses=typed_lenses)

        self.assert_matches_the_filmed_cameras(setup)

    def test_a_typed_focal_length_without_close_ups_is_refined_by_the_camera_pair(self) -> None:
        typed_lenses = (typed_lens_of(fixtures.FRONT_CAMERA, 0.03, False), typed_lens_of(fixtures.SIDE_CAMERA, -0.03, False))

        setup = self.calibrate(self.sightings_outside(0.0, 2.0 * fixtures.CLOSE_UP_SECONDS), typed_lenses=typed_lenses)

        self.assert_matches_the_filmed_cameras(setup)

    def test_a_saved_lens_for_another_video_size_is_refused(self) -> None:
        front_lens, _side_lens = self.saved_lenses_from_a_full_calibration()
        portrait_lens = replace(front_lens, image_width=fixtures.HEIGHT, image_height=fixtures.WIDTH)

        with self.assertRaisesRegex(CalibrationError, "saved lens Front"):
            self.calibrate([self.front_sightings, self.side_sightings], saved_lenses=(portrait_lens, None))

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
        self.assertEqual(loaded.inputs, setup.inputs)
        self.assertIsNotNone(find_current_setup(path, calibration_inputs(self.video_paths)))
        self.assertIsNone(find_current_setup(path, replace(calibration_inputs(self.video_paths), square_size_metres=0.05)))
        self.assertIsNone(find_current_setup(path, replace(calibration_inputs(self.video_paths), stop_frame=100)))
        with_saved_lens = calibration_inputs(self.video_paths, saved_lenses=(saved_lens_of(setup.cameras[0], "Front"), None))
        self.assertIsNone(find_current_setup(path, with_saved_lens))
        with_typed_lens = calibration_inputs(self.video_paths, typed_lenses=(None, TypedLens(23.0, True)))
        self.assertIsNone(find_current_setup(path, with_typed_lens))
        self.assertAlmostEqual(loaded.cameras[0].focal_length_millimetres, setup.cameras[0].focal_length_millimetres)
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


class SavedLensTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_folder = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary_folder.name)
        self.lens = SavedLens.from_arrays(
            "Pixel 8 main 1080p",
            1920,
            1080,
            np.array([[1400.5, 0.0, 960.0], [0.0, 1400.5, 540.0], [0.0, 0.0, 1.0]]),
            np.array([0.08, -0.2, 0.0, 0.0, 0.0]),
            0.31,
        )

    def tearDown(self) -> None:
        self.temporary_folder.cleanup()

    def test_a_saved_lens_loads_with_the_same_values(self) -> None:
        path = self.lens.save(self.folder)

        lenses, problems = load_lenses(self.folder)

        self.assertEqual(path.name, "Pixel 8 main 1080p.json")
        self.assertEqual(lenses, [self.lens])
        self.assertEqual(problems, [])

    def test_a_broken_lens_file_is_reported_while_the_others_load(self) -> None:
        self.lens.save(self.folder)
        (self.folder / "Broken.json").write_text('{"format_version": 1, "name": "Broken"}', encoding="utf-8")

        lenses, problems = load_lenses(self.folder)

        self.assertEqual([lens.name for lens in lenses], ["Pixel 8 main 1080p"])
        self.assertEqual(len(problems), 1)
        self.assertIn("Broken.json", problems[0])

    def test_unsafe_names_are_refused_and_deleting_removes_the_file(self) -> None:
        with self.assertRaises(SavedLensError):
            lens_path(self.folder, "../outside")
        path = self.lens.save(self.folder)

        delete_lens(self.folder, self.lens.name)

        self.assertFalse(path.exists())
        with self.assertRaises(SavedLensError):
            delete_lens(self.folder, self.lens.name)


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
