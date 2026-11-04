import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

from prepare_extension_files import MODEL_FILE_NAMES
from repack_mediapipe_wheel import PatchError, verify_wheel

_REPOSITORY_FOLDER = Path(__file__).resolve().parent.parent
_EXTENSION_FOLDER = _REPOSITORY_FOLDER / "src" / "freeclipmocap"
_OUTPUT_FOLDER = _REPOSITORY_FOLDER / "dist"
_TEST_RUNNER = _REPOSITORY_FOLDER / "Tests" / "run_in_blender.py"
_DEFAULT_BLENDER = Path("C:/Program Files/Blender Foundation/Blender 5.2/blender.exe")
_UPLOAD_LIMIT_MEGABYTES = 200.0
_BUILT_IN_PRESETS = ("presets/bone_maps/Mixamo.json", "presets/bone_maps/Rigify.json")
_LICENCE_FILES = ("THIRD_PARTY_NOTICES.md", "LICENSES/Apache-2.0.txt", "LICENSES/GPL-3.0-or-later.txt")
_EXPECTED_PACKAGES = {"mediapipe", "cv2"}


class BuildError(Exception):
    pass


def recreate_output_folder() -> None:
    if _OUTPUT_FOLDER.exists():
        unexpected = [path.name for path in _OUTPUT_FOLDER.iterdir() if path.suffix != ".zip"]
        if unexpected:
            raise BuildError(f"Refusing to clear {_OUTPUT_FOLDER}: it holds files the build did not make: {', '.join(unexpected)}.")
        shutil.rmtree(_OUTPUT_FOLDER)
    _OUTPUT_FOLDER.mkdir()


def run_blender(blender: Path, arguments: list[str], environment: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        [str(blender), *arguments], capture_output=True, text=True, encoding="utf-8", errors="replace", env=environment
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise BuildError(f"Blender failed ({' '.join(arguments[:4])} …):\n{output}")
    return output


def build_zip(blender: Path) -> Path:
    run_blender(
        blender,
        ["--factory-startup", "--command", "extension", "build", "--source-dir", str(_EXTENSION_FOLDER), "--output-dir", str(_OUTPUT_FOLDER)],
    )
    zip_paths = sorted(_OUTPUT_FOLDER.glob("*.zip"))
    if len(zip_paths) != 1:
        raise BuildError(f"Expected one zip in {_OUTPUT_FOLDER}, found {len(zip_paths)}.")
    return zip_paths[0]


def check_zip(zip_path: Path) -> None:
    with open(_EXTENSION_FOLDER / "blender_manifest.toml", "rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    expected = {
        "blender_manifest.toml",
        "__init__.py",
        *(wheel.removeprefix("./") for wheel in manifest["wheels"]),
        *(f"models/{file_name}" for file_name in MODEL_FILE_NAMES),
        *_BUILT_IN_PRESETS,
        *_LICENCE_FILES,
    }
    missing = sorted(expected - names)
    if missing:
        raise BuildError(f"Missing in {zip_path.name}: {', '.join(missing)}.")
    repository_licence = (_REPOSITORY_FOLDER / "LICENSE").read_text(encoding="utf-8")
    extension_licence = (_EXTENSION_FOLDER / "LICENSES" / "GPL-3.0-or-later.txt").read_text(encoding="utf-8")
    if repository_licence != extension_licence:
        raise BuildError("LICENSES/GPL-3.0-or-later.txt differs from the repository's LICENSE.")
    mediapipe_wheels = [wheel for wheel in manifest["wheels"] if Path(wheel).name.startswith("mediapipe-")]
    if len(mediapipe_wheels) != 1:
        raise BuildError(f"Expected one mediapipe wheel in the manifest, found {len(mediapipe_wheels)}.")
    try:
        verify_wheel(_EXTENSION_FOLDER / mediapipe_wheels[0])
    except PatchError as error:
        raise BuildError(str(error)) from error
    compiled = sorted(name for name in names if "__pycache__" in name or name.endswith(".pyc"))
    if compiled:
        raise BuildError(f"Compiled Python files in {zip_path.name}: {', '.join(compiled[:5])}.")
    megabytes = zip_path.stat().st_size / 1_000_000
    if megabytes > _UPLOAD_LIMIT_MEGABYTES:
        raise BuildError(f"{zip_path.name} is {megabytes:.0f} MB, over the {_UPLOAD_LIMIT_MEGABYTES:.0f} MB upload limit.")
    print(f"Built:     {zip_path.name}, {len(names)} files, {megabytes:.0f} MB (upload limit about {_UPLOAD_LIMIT_MEGABYTES:.0f} MB)")


def test_installed_zip(blender: Path, zip_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="freeclipmocap-install-test-") as user_folder:
        environment = dict(os.environ, BLENDER_USER_RESOURCES=user_folder)
        run_blender(
            blender,
            ["--factory-startup", "--command", "extension", "install-file", "--repo", "user_default", "--enable", str(zip_path)],
            environment,
        )
        extensions_folder = Path(user_folder) / "extensions"
        installed = extensions_folder / "user_default" / "freeclipmocap"
        if installed.is_symlink() or not (installed / "blender_manifest.toml").is_file():
            raise BuildError(f"The zip did not install into {installed}.")
        site_packages = next(extensions_folder.glob(".local/lib/python*/site-packages"), None)
        if site_packages is None:
            raise BuildError("The wheels were not installed.")
        packages = {path.name for path in site_packages.iterdir() if not path.name.endswith(".dist-info") and path.name != "__pycache__"}
        print(f"Installed: into a throwaway Blender user folder; wheel packages: {', '.join(sorted(packages))}")
        if packages != _EXPECTED_PACKAGES:
            raise BuildError(f"Expected only {', '.join(sorted(_EXPECTED_PACKAGES))} from the wheels, found {', '.join(sorted(packages))}.")
        output = run_blender(blender, ["-b", "--factory-startup", "--python-exit-code", "1", "--python", str(_TEST_RUNNER)], environment)
        summary = [line for line in output.splitlines() if line.startswith(("Ran ", "OK", "FAILED"))]
        print(f"Tested:    the installed copy: {' '.join(summary)}")


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description="Build the extension zip into dist/ (deleted and recreated), check it, install it into a "
        "throwaway Blender user folder and run the tests against that copy. Your own Blender setup is not touched."
    )
    argument_parser.add_argument("--blender", type=Path, default=_DEFAULT_BLENDER, help="Path to blender.exe")
    arguments = argument_parser.parse_args()
    try:
        recreate_output_folder()
        zip_path = build_zip(arguments.blender)
        check_zip(zip_path)
        test_installed_zip(arguments.blender, zip_path)
    except BuildError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
