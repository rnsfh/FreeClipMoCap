from pathlib import Path

import bpy

from ..capture.saved_lens import SavedLens, load_lenses

_EXTENSION_PACKAGE = __name__.rsplit(".", 2)[0]
_LENS_FOLDER_NAME = "lenses"


def user_lens_folder() -> Path:
    return Path(bpy.utils.extension_path_user(_EXTENSION_PACKAGE, path=_LENS_FOLDER_NAME, create=True))


def available_lenses() -> list[SavedLens]:
    lenses, _problems = load_lenses(user_lens_folder())
    return lenses


def find_lens(name: str) -> SavedLens | None:
    return next((lens for lens in available_lenses() if lens.name == name), None)
