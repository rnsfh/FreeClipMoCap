import argparse
import os
import subprocess
import sys
from pathlib import Path

_BLENDER_VERSION = "5.2"
_EXTENSION_FOLDER = Path(__file__).resolve().parent.parent / "src" / "freeclipmocap"


def link_path() -> Path:
    blender_user_folder = Path(os.environ["APPDATA"]) / "Blender Foundation" / "Blender" / _BLENDER_VERSION
    return blender_user_folder / "extensions" / "user_default" / _EXTENSION_FOLDER.name


def is_link_to_extension(path: Path) -> bool:
    return path.is_junction() and path.resolve() == _EXTENSION_FOLDER


def link() -> int:
    path = link_path()
    if is_link_to_extension(path):
        print(f"Already linked: {path} -> {_EXTENSION_FOLDER}")
        return 0
    if path.exists() or path.is_junction():
        print(f"{path} exists and is not a link to {_EXTENSION_FOLDER}. Left untouched.", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(path), str(_EXTENSION_FOLDER)], check=True)
    return 0


def unlink() -> int:
    path = link_path()
    if not is_link_to_extension(path):
        print(f"{path} is not a link to {_EXTENSION_FOLDER}. Left untouched.", file=sys.stderr)
        return 1
    os.rmdir(path)
    print(f"Removed the link {path} (the source folder is untouched).")
    return 0


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description="Link src/freeclipmocap into Blender's user_default extensions folder with a directory junction."
    )
    argument_parser.add_argument("--unlink", action="store_true", help="remove the junction instead")
    arguments = argument_parser.parse_args()
    return unlink() if arguments.unlink else link()


if __name__ == "__main__":
    sys.exit(main())
