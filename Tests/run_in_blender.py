import sys
import unittest
from pathlib import Path

import bpy

_MODULE_NAME = "bl_ext.user_default.freeclipmocap"
_TESTS_FOLDER = Path(__file__).resolve().parent


def main() -> int:
    bpy.ops.preferences.addon_enable(module=_MODULE_NAME)
    if "mediapipe" in sys.modules or "cv2" in sys.modules:
        print("FAIL: enabling the extension loaded mediapipe or cv2; they must load only when a capture runs.")
        return 1
    print("Enabling the extension did not load mediapipe or cv2.")
    test_suite = unittest.defaultTestLoader.discover(str(_TESTS_FOLDER), top_level_dir=str(_TESTS_FOLDER.parent))
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
