import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bl_ext.user_default.freeclipmocap.capture.capture_settings import CaptureSettings, HandSearch, PoseModelVariant
from bl_ext.user_default.freeclipmocap.capture.landmark_cache import (
    CacheFormatError,
    HandTrack,
    LandmarkCache,
    PoseTrack,
    VideoSource,
    cache_path_for,
    find_current_cache,
)

_FRAME_COUNT = 4


def make_hand_track(found: list[bool]) -> HandTrack:
    return HandTrack(
        found=np.array(found, dtype=bool),
        image_landmarks=np.random.default_rng(1).random((len(found), 21, 3), dtype=np.float32),
        world_landmarks=np.random.default_rng(2).random((len(found), 21, 3), dtype=np.float32),
        score=np.linspace(0.5, 0.9, len(found), dtype=np.float32),
    )


def make_cache(settings: CaptureSettings, video_path: Path) -> LandmarkCache:
    status = video_path.stat()
    image_landmarks = np.random.default_rng(3).random((_FRAME_COUNT, 33, 3), dtype=np.float32)
    image_landmarks[1] = np.nan
    return LandmarkCache(
        tracking_model="mediapipe-pose33-hands21",
        settings=settings,
        video=VideoSource(
            file_size=status.st_size,
            modified_time_nanoseconds=status.st_mtime_ns,
            frame_rate=59.94,
            width=1920,
            height=1080,
        ),
        frame_indices=np.arange(_FRAME_COUNT, dtype=np.int32),
        timestamps_milliseconds=np.array([0, 17, 33, 50], dtype=np.int64),
        pose=PoseTrack(
            found=np.array([True, False, True, True]),
            image_landmarks=image_landmarks,
            world_landmarks=np.random.default_rng(4).random((_FRAME_COUNT, 33, 3), dtype=np.float32),
            visibility=np.random.default_rng(5).random((_FRAME_COUNT, 33), dtype=np.float32),
            presence=np.random.default_rng(6).random((_FRAME_COUNT, 33), dtype=np.float32),
        ),
        left_hand=make_hand_track([True, False, False, True]),
        right_hand=make_hand_track([False, False, True, True]),
    )


class LandmarkCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_folder = tempfile.TemporaryDirectory()
        self.video_path = Path(self.temporary_folder.name) / "clip.mp4"
        self.video_path.write_bytes(b"not really a video")
        self.settings = CaptureSettings(pose_model_variant=PoseModelVariant.HEAVY)

    def tearDown(self) -> None:
        self.temporary_folder.cleanup()

    def test_cache_path_is_in_a_folder_next_to_the_video(self) -> None:
        path = cache_path_for(self.video_path, self.settings)
        self.assertEqual(path.parent, self.video_path.parent / "FreeClipMocap")
        self.assertTrue(path.name.startswith("clip.mp4.heavy-"))
        self.assertTrue(path.name.endswith(".landmarks.npz"))

    def test_each_pose_model_gets_its_own_cache_path(self) -> None:
        paths = {
            cache_path_for(self.video_path, CaptureSettings(pose_model_variant=variant)) for variant in PoseModelVariant
        }
        self.assertEqual(len(paths), len(PoseModelVariant))

    def test_save_and_load_keep_every_value(self) -> None:
        cache = make_cache(self.settings, self.video_path)
        path = cache_path_for(self.video_path, self.settings)
        cache.save(path)
        loaded = LandmarkCache.load(path)
        self.assertEqual(loaded.tracking_model, cache.tracking_model)
        self.assertEqual(loaded.settings, cache.settings)
        self.assertEqual(loaded.video, cache.video)
        np.testing.assert_array_equal(loaded.frame_indices, cache.frame_indices)
        np.testing.assert_array_equal(loaded.timestamps_milliseconds, cache.timestamps_milliseconds)
        for name in ("found", "image_landmarks", "world_landmarks", "visibility", "presence"):
            np.testing.assert_array_equal(getattr(loaded.pose, name), getattr(cache.pose, name))
        for side in ("left_hand", "right_hand"):
            for name in ("found", "image_landmarks", "world_landmarks", "score"):
                np.testing.assert_array_equal(getattr(getattr(loaded, side), name), getattr(getattr(cache, side), name))
        self.assertTrue(np.isnan(loaded.pose.image_landmarks[1]).all())

    def test_caches_from_full_frame_hand_search_are_out_of_date(self) -> None:
        old_settings = CaptureSettings.from_json('{"pose_model_variant": "heavy"}')
        self.assertEqual(old_settings.hand_search, HandSearch.FULL_FRAME)
        self.assertEqual(self.settings.hand_search, HandSearch.WRIST_CROPS)
        self.assertNotEqual(cache_path_for(self.video_path, old_settings), cache_path_for(self.video_path, self.settings))
        self.assertEqual(CaptureSettings.from_json(self.settings.to_json()), self.settings)

    def test_sliced_cache_keeps_only_the_chosen_frames(self) -> None:
        cache = make_cache(self.settings, self.video_path)
        sliced = cache.sliced(1, 3)
        np.testing.assert_array_equal(sliced.frame_indices, [1, 2])
        np.testing.assert_array_equal(sliced.timestamps_milliseconds, [17, 33])
        np.testing.assert_array_equal(sliced.pose.found, [False, True])
        np.testing.assert_array_equal(sliced.pose.visibility, cache.pose.visibility[1:3])
        np.testing.assert_array_equal(sliced.left_hand.found, [False, False])
        np.testing.assert_array_equal(sliced.right_hand.score, cache.right_hand.score[1:3])
        self.assertEqual(sliced.video, cache.video)

    def test_loaded_arrays_are_read_only(self) -> None:
        path = cache_path_for(self.video_path, self.settings)
        make_cache(self.settings, self.video_path).save(path)
        loaded = LandmarkCache.load(path)
        with self.assertRaises(ValueError):
            loaded.pose.world_landmarks[0, 0, 0] = 1.0

    def test_cache_is_current_until_the_video_changes(self) -> None:
        path = cache_path_for(self.video_path, self.settings)
        make_cache(self.settings, self.video_path).save(path)
        self.assertEqual(find_current_cache(self.video_path, self.settings), path)
        status = self.video_path.stat()
        os.utime(self.video_path, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000))
        self.assertIsNone(find_current_cache(self.video_path, self.settings))

    def test_no_cache_for_other_settings(self) -> None:
        make_cache(self.settings, self.video_path).save(cache_path_for(self.video_path, self.settings))
        preview_settings = CaptureSettings(pose_model_variant=PoseModelVariant.LITE)
        self.assertIsNone(find_current_cache(self.video_path, preview_settings))

    def test_other_format_version_is_refused(self) -> None:
        path = cache_path_for(self.video_path, self.settings)
        make_cache(self.settings, self.video_path).save(path)
        with np.load(path, allow_pickle=False) as arrays:
            values = {name: arrays[name] for name in arrays.files}
        values["format_version"] = np.array(99, dtype=np.int32)
        with open(path, "wb") as cache_file:
            np.savez(cache_file, **values)
        with self.assertRaises(CacheFormatError):
            LandmarkCache.load(path)
        self.assertIsNone(find_current_cache(self.video_path, self.settings))


if __name__ == "__main__":
    unittest.main()
