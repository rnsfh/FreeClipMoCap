import argparse
import shutil
import sys
import tomllib
from pathlib import Path

from repack_mediapipe_wheel import PatchError, print_result, repack_mediapipe_wheel

_EXTENSION_FOLDER = Path(__file__).resolve().parent.parent / "src" / "freeclipmocap"
_MEDIAPIPE_WHEEL_PATTERN = "mediapipe-*-win_amd64.whl"
_OPENCV_WHEEL_PATTERN = "opencv_contrib_python-*-win_amd64.whl"
MODEL_FILE_NAMES = (
    "pose_landmarker_lite.task",
    "pose_landmarker_full.task",
    "pose_landmarker_heavy.task",
    "hand_landmarker.task",
)


class PreparationError(Exception):
    pass


def find_single_wheel(source_folder: Path, pattern: str) -> Path:
    matches = sorted(source_folder.glob(pattern))
    if len(matches) != 1:
        raise PreparationError(f"Expected exactly one {pattern} in {source_folder}, found {len(matches)}.")
    return matches[0]


def recreate_generated_folder(folder: Path) -> None:
    if folder.parent != _EXTENSION_FOLDER:
        raise PreparationError(f"Refusing to clear {folder}: not a folder of the extension.")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()


def check_manifest_lists(wheel_file_names: list[str]) -> None:
    with open(_EXTENSION_FOLDER / "blender_manifest.toml", "rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    listed_wheels = sorted(manifest.get("wheels", []))
    expected_wheels = sorted(f"./wheels/{file_name}" for file_name in wheel_file_names)
    if listed_wheels != expected_wheels:
        raise PreparationError(
            "blender_manifest.toml lists other wheels than were prepared. Expected:\n"
            + "\n".join(f'  "{wheel}",' for wheel in expected_wheels)
        )


def prepare_extension_files(source_wheels_folder: Path, source_models_folder: Path) -> None:
    mediapipe_wheel_path = find_single_wheel(source_wheels_folder, _MEDIAPIPE_WHEEL_PATTERN)
    opencv_wheel_path = find_single_wheel(source_wheels_folder, _OPENCV_WHEEL_PATTERN)
    missing_models = [name for name in MODEL_FILE_NAMES if not (source_models_folder / name).is_file()]
    if missing_models:
        raise PreparationError(f"Missing in {source_models_folder}: {', '.join(missing_models)}.")

    wheels_folder = _EXTENSION_FOLDER / "wheels"
    models_folder = _EXTENSION_FOLDER / "models"
    recreate_generated_folder(wheels_folder)
    recreate_generated_folder(models_folder)

    print_result(repack_mediapipe_wheel(mediapipe_wheel_path, wheels_folder))
    shutil.copy2(opencv_wheel_path, wheels_folder / opencv_wheel_path.name)
    print(f"Copied:                  {opencv_wheel_path.name}")
    for model_file_name in MODEL_FILE_NAMES:
        shutil.copy2(source_models_folder / model_file_name, models_folder / model_file_name)
        print(f"Copied:                  {model_file_name}")

    check_manifest_lists([mediapipe_wheel_path.name, opencv_wheel_path.name])
    print("blender_manifest.toml lists the prepared wheels.")


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description="Fill src/freeclipmocap/wheels/ (patched mediapipe, opencv) and models/ (.task files). "
        "Both folders are deleted and recreated."
    )
    argument_parser.add_argument("source_wheels_folder", type=Path)
    argument_parser.add_argument("source_models_folder", type=Path)
    arguments = argument_parser.parse_args()
    try:
        prepare_extension_files(arguments.source_wheels_folder.resolve(), arguments.source_models_folder.resolve())
    except (PatchError, PreparationError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
