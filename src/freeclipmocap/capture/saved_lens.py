import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants.user_files import VALID_NAME_PATTERN

_FORMAT_VERSION = 1
_VALID_NAME_PATTERN = re.compile(VALID_NAME_PATTERN)


class SavedLensError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class SavedLens:
    name: str
    image_width: int
    image_height: int
    camera_matrix_values: tuple[float, ...]
    distortion_values: tuple[float, ...]
    error_pixels: float

    @classmethod
    def from_arrays(
        cls,
        name: str,
        image_width: int,
        image_height: int,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        error_pixels: float,
    ) -> "SavedLens":
        return cls(
            name=name,
            image_width=int(image_width),
            image_height=int(image_height),
            camera_matrix_values=tuple(float(value) for value in np.asarray(camera_matrix).ravel()),
            distortion_values=tuple(float(value) for value in np.asarray(distortion).ravel()),
            error_pixels=float(error_pixels),
        )

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(self.camera_matrix_values, dtype=np.float64).reshape(3, 3)

    @property
    def distortion(self) -> np.ndarray:
        return np.array(self.distortion_values, dtype=np.float64)

    def to_values(self) -> dict[str, object]:
        return {
            "name": self.name,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion": list(self.distortion_values),
            "error_pixels": self.error_pixels,
        }

    @classmethod
    def from_values(cls, values: dict[str, object]) -> "SavedLens":
        camera_matrix = np.array(values["camera_matrix"], dtype=np.float64)
        distortion = np.array(values["distortion"], dtype=np.float64)
        if camera_matrix.shape != (3, 3) or distortion.shape != (5,):
            raise ValueError("a lens needs a 3×3 camera matrix and 5 distortion values")
        return cls.from_arrays(
            str(values["name"]),
            int(values["image_width"]),
            int(values["image_height"]),
            camera_matrix,
            distortion,
            float(values["error_pixels"]),
        )

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = lens_path(folder, self.name)
        document = {"format_version": _FORMAT_VERSION, **self.to_values()}
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "SavedLens":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SavedLensError(f"Cannot read the saved lens {path}: {error}") from error
        if not isinstance(document, dict) or document.get("format_version") != _FORMAT_VERSION:
            raise SavedLensError(f"{path} is not a saved lens of format version {_FORMAT_VERSION}.")
        try:
            return cls.from_values(document)
        except (KeyError, TypeError, ValueError) as error:
            raise SavedLensError(f"{path} is not a valid saved lens: {error}") from error


def is_valid_lens_name(name: str) -> bool:
    return _VALID_NAME_PATTERN.fullmatch(name) is not None


def lens_path(folder: Path, name: str) -> Path:
    if not is_valid_lens_name(name):
        raise SavedLensError(f"Lens names may use letters, digits, spaces, '-' and '_' only: {name!r}.")
    return folder / f"{name}.json"


def load_lenses(folder: Path) -> tuple[list[SavedLens], list[str]]:
    lenses: list[SavedLens] = []
    problems: list[str] = []
    if not folder.is_dir():
        return lenses, problems
    for path in sorted(folder.glob("*.json")):
        try:
            lenses.append(SavedLens.load(path))
        except SavedLensError as error:
            problems.append(str(error))
    return lenses, problems


def delete_lens(folder: Path, name: str) -> None:
    path = lens_path(folder, name)
    if not path.is_file():
        raise SavedLensError(f"The saved lens {name} is not in {folder}.")
    path.unlink()
