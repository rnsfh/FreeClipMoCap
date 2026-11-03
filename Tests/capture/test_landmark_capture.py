import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from bl_ext.user_default.freeclipmocap.capture.capture_job import CaptureJob
from bl_ext.user_default.freeclipmocap.capture.capture_progress import CaptureProgress
from bl_ext.user_default.freeclipmocap.capture.capture_settings import CaptureSettings, PoseModelVariant
from bl_ext.user_default.freeclipmocap.capture.landmark_cache import LandmarkCache, cache_path_for
from bl_ext.user_default.freeclipmocap.capture.hand_crops import SMALLEST_CROP_PIXELS, CropBox
from bl_ext.user_default.freeclipmocap.capture.landmark_capture import capture_landmarks, hand_crop_box, nearest_hand

from .video_fixtures import write_moving_shapes_video

_MODELS_FOLDER = Path(sys.modules["bl_ext.user_default.freeclipmocap"].__file__).parent / "models"
_MAXIMUM_MAIN_THREAD_PAUSE_SECONDS = 0.1


class HandCropTests(unittest.TestCase):
    def test_nearest_hand_wins_and_far_hands_are_refused(self) -> None:
        expected = np.array([0.5, 0.55])
        self.assertEqual(nearest_hand([np.array([0.9, 0.9]), np.array([0.52, 0.5])], expected), 1)
        self.assertIsNone(nearest_hand([np.array([0.7, 0.55])], expected))
        self.assertIsNone(nearest_hand([], expected))

    def test_box_is_centred_between_wrist_and_knuckles_and_never_tiny(self) -> None:
        box = CropBox.around_hand(np.array([100.0, 200.0]), np.array([100.0, 240.0]), torso_pixels=100.0)
        self.assertAlmostEqual(box.side, 600.0)
        self.assertAlmostEqual(box.left + box.side / 2, 100.0)
        self.assertAlmostEqual(box.top + box.side / 2, 220.0)
        pointing_at_camera = CropBox.around_hand(np.array([100.0, 200.0]), np.array([100.0, 202.0]), torso_pixels=300.0)
        self.assertAlmostEqual(pointing_at_camera.side, 660.0)
        self.assertAlmostEqual(CropBox.around_hand(np.zeros(2), np.zeros(2), torso_pixels=10.0).side, 96.0)

    def test_crop_is_enlarged_and_maps_back_to_the_frame(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:110, 600:610] = 255
        box = CropBox(left=560.0, top=60.0, side=128.0)
        crop = box.cut(frame)
        self.assertEqual(crop.shape, (SMALLEST_CROP_PIXELS, SMALLEST_CROP_PIXELS, 3))
        self.assertEqual(CropBox(left=0.0, top=0.0, side=400.4).cut(frame).shape, (401, 401, 3))
        white = np.argwhere(crop[..., 0] > 128)
        self.assertAlmostEqual(float(white[:, 1].mean()), (605.0 - 560.0) * 2.0, delta=1.5)
        self.assertEqual(int(crop[:, -1].max()), 0)
        crop_point = np.array([[white[:, 1].mean() / SMALLEST_CROP_PIXELS, white[:, 0].mean() / SMALLEST_CROP_PIXELS, 0.1]])
        frame_point = box.to_frame(crop_point, 640, 480)[0]
        self.assertAlmostEqual(float(frame_point[0] * 640), 604.5, delta=1.0)
        self.assertAlmostEqual(float(frame_point[1] * 480), 104.5, delta=1.0)
        self.assertAlmostEqual(float(frame_point[2]), 0.1 * 128.0 / 640.0)

    def test_no_crop_for_a_hidden_wrist(self) -> None:
        pose_pixels = np.zeros((33, 2))
        visibility = np.ones(33)
        visibility[15] = 0.1
        self.assertIsNone(hand_crop_box(pose_pixels, visibility, "left"))
        self.assertIsNotNone(hand_crop_box(pose_pixels, visibility, "right"))


class CaptureLandmarksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_folder = tempfile.TemporaryDirectory()
        cls.video_path = write_moving_shapes_video(Path(cls.temporary_folder.name) / "shapes.mp4", frame_count=30)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_folder.cleanup()

    def test_every_frame_is_recorded_and_missing_pose_is_nan(self) -> None:
        settings = CaptureSettings(pose_model_variant=PoseModelVariant.LITE)
        progress = CaptureProgress()
        cache = capture_landmarks(self.video_path, settings, _MODELS_FOLDER, progress, threading.Event())
        self.assertEqual(cache.frame_count, 30)
        self.assertEqual(progress.frames_done, 30)
        self.assertEqual((cache.video.width, cache.video.height), (640, 480))
        self.assertAlmostEqual(cache.video.frame_rate, 30.0, places=1)
        self.assertTrue(np.all(np.diff(cache.timestamps_milliseconds) > 0))
        self.assertEqual(cache.pose.image_landmarks.shape, (30, 33, 3))
        self.assertEqual(cache.pose.world_landmarks.dtype, np.float32)
        self.assertEqual(cache.left_hand.image_landmarks.shape, (30, 21, 3))
        self.assertFalse(cache.pose.found.any())
        self.assertTrue(np.isnan(cache.pose.world_landmarks).all())
        self.assertFalse(cache.left_hand.found.any() or cache.right_hand.found.any())


class CaptureJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_folder = tempfile.TemporaryDirectory()
        cls.video_path = write_moving_shapes_video(Path(cls.temporary_folder.name) / "long.mp4", frame_count=300)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_folder.cleanup()

    def test_main_thread_stays_responsive_and_cache_is_written(self) -> None:
        settings = CaptureSettings(pose_model_variant=PoseModelVariant.HEAVY)
        cache_path = cache_path_for(self.video_path, settings)
        job = CaptureJob(self.video_path, settings, _MODELS_FOLDER, cache_path)
        started = time.perf_counter()
        job.start()
        longest_pause_seconds = 0.0
        main_thread_ticks = 0
        while job.is_running:
            before_sleep = time.perf_counter()
            time.sleep(0.01)
            longest_pause_seconds = max(longest_pause_seconds, time.perf_counter() - before_sleep - 0.01)
            main_thread_ticks += 1
        elapsed_seconds = time.perf_counter() - started
        print(
            f"\n  capture of 300 frames (heavy) took {elapsed_seconds:.1f} s; main thread ran {main_thread_ticks} "
            f"ticks, longest extra pause {longest_pause_seconds * 1000:.0f} ms",
            file=sys.stderr,
        )
        self.assertIsNone(job.error_message)
        self.assertIsNotNone(job.cache)
        self.assertEqual(LandmarkCache.load(cache_path).frame_count, 300)
        self.assertLess(longest_pause_seconds, _MAXIMUM_MAIN_THREAD_PAUSE_SECONDS)

    def test_cancel_stops_the_capture_and_writes_no_cache(self) -> None:
        settings = CaptureSettings(pose_model_variant=PoseModelVariant.FULL)
        cache_path = cache_path_for(self.video_path, settings)
        job = CaptureJob(self.video_path, settings, _MODELS_FOLDER, cache_path)
        job.start()
        while job.progress.frames_done < 5 and job.is_running:
            time.sleep(0.01)
        job.cancel()
        job.wait(30.0)
        self.assertFalse(job.is_running)
        self.assertTrue(job.was_cancelled)
        self.assertIsNone(job.cache)
        self.assertFalse(cache_path.exists())

    def test_unreadable_video_reports_an_error(self) -> None:
        broken_video_path = Path(self.temporary_folder.name) / "broken.mp4"
        broken_video_path.write_bytes(b"not a video")
        settings = CaptureSettings(pose_model_variant=PoseModelVariant.LITE)
        job = CaptureJob(broken_video_path, settings, _MODELS_FOLDER, cache_path_for(broken_video_path, settings))
        job.start()
        job.wait(30.0)
        self.assertIsNotNone(job.error_message)
        self.assertIsNone(job.cache)


if __name__ == "__main__":
    unittest.main()
