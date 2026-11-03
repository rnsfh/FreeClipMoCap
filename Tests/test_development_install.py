import importlib
import sys
import unittest
from pathlib import Path

import bpy

_MODULE_NAME = "bl_ext.user_default.freeclipmocap"
_BLOCKED_ENDPOINT = b"https://127.0.0.1:1/blocked0000"
_TELEMETRY_ENDPOINT = b"https://play.googleapis.com/log"
_MODEL_FILE_NAMES = (
    "pose_landmarker_lite.task",
    "pose_landmarker_full.task",
    "pose_landmarker_heavy.task",
    "hand_landmarker.task",
)


class DevelopmentInstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bpy.ops.preferences.addon_enable(module=_MODULE_NAME)

    def test_extension_is_enabled(self) -> None:
        self.assertIn(_MODULE_NAME, bpy.context.preferences.addons)

    def test_panel_is_registered(self) -> None:
        self.assertTrue(hasattr(bpy.types, "FREECLIPMOCAP_PT_main"))

    def test_mediapipe_comes_from_extension_wheels(self) -> None:
        mediapipe = importlib.import_module("mediapipe")
        self.assertIn(".local", Path(mediapipe.__file__).parts)

    def test_opencv_comes_from_extension_wheels(self) -> None:
        cv2 = importlib.import_module("cv2")
        self.assertIn(".local", Path(cv2.__file__).parts)

    def test_telemetry_endpoint_is_blocked(self) -> None:
        mediapipe = importlib.import_module("mediapipe")
        library = (Path(mediapipe.__file__).parent / "tasks" / "c" / "libmediapipe.dll").read_bytes()
        self.assertEqual(library.count(_TELEMETRY_ENDPOINT), 0)
        self.assertEqual(library.count(_BLOCKED_ENDPOINT), 1)

    def test_vision_loads_without_matplotlib(self) -> None:
        vision = importlib.import_module("mediapipe.tasks.python.vision")
        self.assertTrue(hasattr(vision, "PoseLandmarker"))
        self.assertNotIn("matplotlib", sys.modules)

    def test_models_are_bundled(self) -> None:
        extension = sys.modules[_MODULE_NAME]
        models_folder = Path(extension.__file__).parent / "models"
        for model_file_name in _MODEL_FILE_NAMES:
            self.assertTrue((models_folder / model_file_name).is_file(), model_file_name)


if __name__ == "__main__":
    test_program = unittest.main(argv=[sys.argv[0]], exit=False, verbosity=2)
    if not test_program.result.wasSuccessful():
        sys.exit(1)
