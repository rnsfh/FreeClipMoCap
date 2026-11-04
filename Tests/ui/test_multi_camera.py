import tempfile
import unittest
import wave
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from bl_ext.user_default.freeclipmocap.capture.calibration_job import CalibrationJob
from bl_ext.user_default.freeclipmocap.capture.camera_setup import calibration_path_for
from bl_ext.user_default.freeclipmocap.capture.landmark_cache import cache_path_for
from bl_ext.user_default.freeclipmocap.constants.camera_lens import FULL_FRAME_WIDTH_MILLIMETRES
from bl_ext.user_default.freeclipmocap.capture.saved_lens import SavedLens, lens_path
from bl_ext.user_default.freeclipmocap.ui.calibration_operators import (
    CalibrationRequestError,
    apply_camera_setup,
    calibration_request,
    calibration_state,
    describe_setup,
    reuse_saved_calibration,
)
from bl_ext.user_default.freeclipmocap.ui.capture_operators import MODELS_FOLDER
from bl_ext.user_default.freeclipmocap.ui.lens_folder import user_lens_folder
from bl_ext.user_default.freeclipmocap.capture.camera_setup import TypedLens
from bl_ext.user_default.freeclipmocap.ui.properties import MEASURE_LENS_IDENTIFIER, TYPED_LENS_IDENTIFIER
from bl_ext.user_default.freeclipmocap.ui.phone_cameras import phone_camera_name

from ..animation.test_action_writer import create_armature
from ..capture import body_fixtures
from ..capture import calibration_fixtures as fixtures
from ..solver.synthetic_actor import make_rig
from ..solver.test_audio_sync import room_sound

_POINTS = np.array([[0.0, 0.0, 1.0], [0.3, 0.2, 0.2], [-0.2, -0.1, 1.5], [0.1, 0.3, 0.8]])
_TEST_LENS_NAMES = ("FreeClipMocap Test Lens", "FreeClipMocap Test Lens A", "FreeClipMocap Test Lens B")


def made_up_lens(name: str) -> SavedLens:
    camera_matrix = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
    return SavedLens.from_arrays(name, 1920, 1080, camera_matrix, np.zeros(5), 0.3)


def write_sound(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    with wave.open(str(path), "wb") as sound_file:
        sound_file.setnchannels(1)
        sound_file.setsampwidth(2)
        sound_file.setframerate(sample_rate)
        sound_file.writeframes((np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    return path


def blender_pixels(scene: bpy.types.Scene, camera_object: bpy.types.Object, points: np.ndarray) -> np.ndarray:
    width, height = scene.render.resolution_x, scene.render.resolution_y
    pixels = []
    for point in points:
        normalized = world_to_camera_view(scene, camera_object, Vector(point))
        pixels.append((normalized.x * width, (1.0 - normalized.y) * height))
    return np.array(pixels)


class MultiCameraTests(unittest.TestCase):
    def setUp(self) -> None:
        bpy.ops.wm.read_homefile(use_empty=True)
        self.properties = bpy.context.scene.freeclipmocap
        self.temporary_folder = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary_folder.name)

    def tearDown(self) -> None:
        calibration_state.set_job(None)
        calibration_state.set_setup(None)
        calibration_state.set_status_message("")
        self.temporary_folder.cleanup()
        for name in _TEST_LENS_NAMES:
            lens_path(user_lens_folder(), name).unlink(missing_ok=True)

    def test_cameras_are_added_and_removed(self) -> None:
        bpy.ops.freeclipmocap.add_camera_video()
        bpy.ops.freeclipmocap.add_camera_video()
        self.properties.extra_videos[1].video_path = "kept.mp4"

        bpy.ops.freeclipmocap.remove_camera_video(index=0)

        self.assertEqual([camera.video_path for camera in self.properties.extra_videos], ["kept.mp4"])

    def test_sync_by_sound_fills_in_how_much_later_each_camera_started(self) -> None:
        events = [1.1, 2.4, 3.05, 5.2, 6.7]
        later_by = 0.4213
        first = write_sound(self.folder / "first.wav", room_sound(8.0, 48000.0, events, seed=7), 48000)
        second = write_sound(self.folder / "second.wav", room_sound(8.0, 44100.0, [time - later_by for time in events], seed=8), 44100)
        self.properties.video_path = str(first)
        self.properties.extra_videos.add().video_path = str(second)

        bpy.ops.freeclipmocap.sync_by_sound()

        self.assertAlmostEqual(self.properties.extra_videos[0].started_later_seconds, later_by, delta=0.002)

    def test_the_distance_below_the_squares_comes_from_the_paper_or_is_entered(self) -> None:
        self.properties.board_square_millimetres = 48.0

        self.properties.paper_size = "A4"
        on_a4 = self.properties.edge_below_squares_metres()
        self.properties.paper_size = "LETTER"
        on_letter = self.properties.edge_below_squares_metres()
        self.properties.paper_size = "CUSTOM"
        self.properties.board_edge_below_squares_millimetres = 14.0
        on_a_tablet = self.properties.edge_below_squares_metres()

        self.assertAlmostEqual(on_a4, 0.0285)
        self.assertAlmostEqual(on_letter, 0.0197)
        self.assertAlmostEqual(on_a_tablet, 0.014)

    def test_the_lens_menu_keeps_its_lens_by_name_and_shows_a_missing_one(self) -> None:
        front_video, side_video = fixtures.shared_calibration_videos()
        self.properties.video_path = str(front_video)
        self.properties.extra_videos.add().video_path = str(side_video)
        made_up_lens("FreeClipMocap Test Lens B").save(user_lens_folder())
        self.properties.calibration_lens = "FreeClipMocap Test Lens B"

        made_up_lens("FreeClipMocap Test Lens A").save(user_lens_folder())
        kept_name = self.properties.calibration_lens
        _video_paths, inputs = calibration_request(self.properties)
        lens_path(user_lens_folder(), "FreeClipMocap Test Lens B").unlink()

        self.assertEqual(kept_name, "FreeClipMocap Test Lens B")
        self.assertEqual(inputs.saved_lenses, (made_up_lens("FreeClipMocap Test Lens B"), None))
        self.assertEqual(self.properties.calibration_lens, "FreeClipMocap Test Lens B")
        with self.assertRaisesRegex(CalibrationRequestError, "Camera 1: the saved lens FreeClipMocap Test Lens B"):
            calibration_request(self.properties)
        self.assertEqual(self.properties.extra_videos[0].calibration_lens, MEASURE_LENS_IDENTIFIER)
        self.properties.calibration_lens = MEASURE_LENS_IDENTIFIER
        self.assertEqual(self.properties.calibration_lens_name, "")

    def test_a_typed_focal_length_goes_into_the_calibration(self) -> None:
        front_video, side_video = fixtures.shared_calibration_videos()
        self.properties.video_path = str(front_video)
        side = self.properties.extra_videos.add()
        side.video_path = str(side_video)
        side.calibration_lens_name = "FreeClipMocap Test Lens"

        side.calibration_lens = TYPED_LENS_IDENTIFIER
        side.calibration_focal_length_millimetres = 23.0
        side.calibration_keep_focal_length = True
        _video_paths, inputs = calibration_request(self.properties)

        self.assertEqual(side.calibration_lens, TYPED_LENS_IDENTIFIER)
        self.assertEqual(side.calibration_lens_name, "")
        self.assertEqual(inputs.typed_lenses, (None, TypedLens(23.0, True)))
        self.assertEqual(inputs.saved_lenses, (None, None))

    def synthetic_take_for_the_body(self) -> None:
        front_video = self.folder / "front.mp4"
        side_video = self.folder / "side.mp4"
        front_video.write_bytes(b"stand-in front video")
        side_video.write_bytes(b"stand-in side video")
        for cache, video_path in zip(body_fixtures.filmed_take(front_video, side_video), (front_video, side_video)):
            cache.save(cache_path_for(video_path, body_fixtures.CAPTURE_SETTINGS))
        self.properties.video_path = str(front_video)
        side = self.properties.extra_videos.add()
        side.video_path = str(side_video)
        side.started_later_seconds = body_fixtures.SIDE_STARTED_LATER_SECONDS
        self.properties.calibration_method = "BODY"

    def type_the_lenses_and_distance(self) -> None:
        for settings, filmed in (
            (self.properties, body_fixtures.FRONT_CAMERA),
            (self.properties.extra_videos[0], body_fixtures.SIDE_CAMERA),
        ):
            settings.calibration_lens = TYPED_LENS_IDENTIFIER
            settings.calibration_focal_length_millimetres = filmed.focal_length_pixels * FULL_FRAME_WIDTH_MILLIMETRES / fixtures.WIDTH
        apart = np.linalg.norm(body_fixtures.SIDE_CAMERA.position - body_fixtures.FRONT_CAMERA.position)
        self.properties.cameras_apart_metres = float(apart)

    def calibrated_job(self) -> CalibrationJob:
        video_paths, inputs = calibration_request(self.properties)
        job = CalibrationJob(video_paths, inputs, calibration_path_for(video_paths[0]), MODELS_FOLDER)
        job.start()
        job.wait(300.0)
        return job

    def test_calibrating_from_the_body_uses_each_video_s_landmarks(self) -> None:
        self.synthetic_take_for_the_body()
        with self.assertRaisesRegex(CalibrationRequestError, "Camera 1: calibrating from the body needs its lens"):
            calibration_request(self.properties)
        self.type_the_lenses_and_distance()

        job = self.calibrated_job()

        self.assertIsNone(job.error_message)
        message = describe_setup(job.setup)
        self.assertIn("floor from the feet", message)
        self.assertIn("mm (typed)", message)
        for calibrated, filmed in zip(job.setup.cameras, (body_fixtures.FRONT_CAMERA, body_fixtures.SIDE_CAMERA)):
            self.assertLess(np.linalg.norm(calibrated.position - filmed.position), 0.04)

    def test_solve_with_all_cameras_triangulates_the_calibrated_take(self) -> None:
        self.synthetic_take_for_the_body()
        self.type_the_lenses_and_distance()
        rig, _bone_map = make_rig()
        bones = [
            (name, None if parent < 0 else rig.bone_names[parent], tuple(head), tuple(tail))
            for name, parent, head, tail in zip(rig.bone_names, rig.parent_indices, rig.heads, rig.tails)
        ]
        self.properties.target_armature = create_armature("TriangulatedRig", bones)
        self.properties.smoothing = 0.5
        self.properties.video_end_frame = 120
        self.assertTrue(self.properties.use_all_cameras)
        with self.assertRaisesRegex(RuntimeError, "not calibrated"):
            bpy.ops.freeclipmocap.solve()

        self.assertIsNone(self.calibrated_job().error_message)
        self.assertEqual(bpy.ops.freeclipmocap.solve(), {"FINISHED"})

        armature_object = self.properties.target_armature
        self.assertEqual(armature_object.animation_data.action.name, "front")
        bpy.context.scene.frame_set(60)
        for side in ("Left", "Right"):
            knee = armature_object.pose.bones[f"mixamorig:{side}Leg"]
            self.assertLess(np.degrees(knee.rotation_quaternion.angle), 12.0, side)

    def test_separate_calibration_videos_must_all_be_chosen(self) -> None:
        front_video, side_video = fixtures.shared_calibration_videos()
        self.properties.video_path = str(front_video)
        self.properties.extra_videos.add().video_path = str(side_video)
        self.properties.use_separate_calibration_videos = True
        self.properties.calibration_video_path = str(front_video)

        with self.assertRaisesRegex(CalibrationRequestError, "Camera 2"):
            calibration_request(self.properties)

    def test_calibrating_the_take_adds_a_camera_per_phone_and_a_second_press_reuses_it(self) -> None:
        front_video, side_video = fixtures.shared_calibration_videos()
        self.properties.video_path = str(front_video)
        side = self.properties.extra_videos.add()
        side.video_path = str(side_video)
        side.started_later_seconds = fixtures.SIDE_STARTED_LATER_SECONDS
        self.properties.board_square_millimetres = fixtures.SQUARE_SIZE_METRES * 1000.0
        video_paths, inputs = calibration_request(self.properties)
        job = CalibrationJob(video_paths, inputs, calibration_path_for(video_paths[0]))

        job.start()
        job.wait(300.0)
        self.assertIsNone(job.error_message)
        apply_camera_setup(bpy.context, job.setup)

        scene = bpy.context.scene
        self.assertTrue(calibration_path_for(front_video).is_file())
        self.assertEqual((scene.render.resolution_x, scene.render.resolution_y), (fixtures.WIDTH, fixtures.HEIGHT))
        for camera_index, (calibrated, filmed) in enumerate(zip(job.setup.cameras, (fixtures.FRONT_CAMERA, fixtures.SIDE_CAMERA))):
            camera_object = scene.objects[phone_camera_name(camera_index)]
            self.assertLess(np.linalg.norm(np.array(camera_object.matrix_world.translation) - filmed.position), 0.03)
            np.testing.assert_allclose(blender_pixels(scene, camera_object, _POINTS), calibrated.project(_POINTS), atol=2.0)
        side_clip = scene.objects[phone_camera_name(1)].data.background_images[0].clip
        self.assertEqual(side_clip.frame_start, 1 + round(fixtures.SIDE_STARTED_LATER_SECONDS * fixtures.FRAME_RATE))

        message = reuse_saved_calibration(bpy.context, *calibration_request(self.properties))

        self.assertIsNotNone(message)
        self.assertTrue(calibration_state.status_message.startswith("Reused the saved calibration."))
        side.started_later_seconds += 0.01
        self.assertIsNone(reuse_saved_calibration(bpy.context, *calibration_request(self.properties)))
        side.started_later_seconds -= 0.01
        side.calibration_lens = TYPED_LENS_IDENTIFIER

        bpy.ops.freeclipmocap.save_lens(camera_index=1, lens_name="FreeClipMocap Test Lens")

        self.assertEqual(side.calibration_lens, "FreeClipMocap Test Lens")
        self.assertIsNone(calibration_request(self.properties)[1].typed_lenses[1])
        _video_paths, inputs = calibration_request(self.properties)
        np.testing.assert_array_equal(inputs.saved_lenses[1].camera_matrix, job.setup.cameras[1].camera_matrix)
        self.assertIsNone(reuse_saved_calibration(bpy.context, *calibration_request(self.properties)))
        self.properties.calibration_lens_name = "FreeClipMocap Test Lens"

        bpy.ops.freeclipmocap.delete_lens(camera_index=1)

        self.assertFalse(lens_path(user_lens_folder(), "FreeClipMocap Test Lens").exists())
        self.assertEqual((self.properties.calibration_lens_name, side.calibration_lens_name), ("", ""))


if __name__ == "__main__":
    unittest.main()
