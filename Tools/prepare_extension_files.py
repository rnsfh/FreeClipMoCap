import argparse
import shutil
import sys
import tomllib
from pathlib import Path

from bundled_wheels import PYPI_SHA256_BY_WHEEL, WheelError, verify_unchanged_wheel

_EXTENSION_FOLDER = Path(__file__).resolve().parent.parent / "src" / "freeclipmocap"
MODEL_FILE_NAMES = (
    "pose_landmarker_lite.task",
    "pose_landmarker_full.task",
    "pose_landmarker_heavy.task",
    "hand_landmarker.task",
)


class PreparationError(Exception):
    pass


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
    wheel_file_names = list(PYPI_SHA256_BY_WHEEL)
    missing_wheels = [name for name in wheel_file_names if not (source_wheels_folder / name).is_file()]
    if missing_wheels:
        raise PreparationError(f"Missing in {source_wheels_folder}: {', '.join(missing_wheels)}.")
    for wheel_file_name in wheel_file_names:
        verify_unchanged_wheel(source_wheels_folder / wheel_file_name)
    missing_models = [name for name in MODEL_FILE_NAMES if not (source_models_folder / name).is_file()]
    if missing_models:
        raise PreparationError(f"Missing in {source_models_folder}: {', '.join(missing_models)}.")
    check_manifest_lists(wheel_file_names)

    wheels_folder = _EXTENSION_FOLDER / "wheels"
    models_folder = _EXTENSION_FOLDER / "models"
    recreate_generated_folder(wheels_folder)
    recreate_generated_folder(models_folder)

    for wheel_file_name in wheel_file_names:
        shutil.copy2(source_wheels_folder / wheel_file_name, wheels_folder / wheel_file_name)
        print(f"Copied (as on PyPI): {wheel_file_name}")
    for model_file_name in MODEL_FILE_NAMES:
        shutil.copy2(source_models_folder / model_file_name, models_folder / model_file_name)
        print(f"Copied:              {model_file_name}")
    print("blender_manifest.toml lists the prepared wheels.")


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description="Fill src/freeclipmocap/wheels/ (the wheels listed in Tools/bundled_wheels.py, checked against "
        "PyPI's SHA-256) and models/ (.task files). Both folders are deleted and recreated."
    )
    argument_parser.add_argument("source_wheels_folder", type=Path)
    argument_parser.add_argument("source_models_folder", type=Path)
    arguments = argument_parser.parse_args()
    try:
        prepare_extension_files(arguments.source_wheels_folder.resolve(), arguments.source_models_folder.resolve())
    except (WheelError, PreparationError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
