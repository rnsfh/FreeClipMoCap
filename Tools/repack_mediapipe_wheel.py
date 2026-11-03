import argparse
import base64
import hashlib
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

_LIBRARY_PATH = "mediapipe/tasks/c/libmediapipe.dll"
_DRAWING_UTILS_PATH = "mediapipe/tasks/python/vision/drawing_utils.py"
_TELEMETRY_ENDPOINT = b"https://play.googleapis.com/log"
_BLOCKED_ENDPOINT = b"https://127.0.0.1:1/blocked0000"
_TOP_LEVEL_MATPLOTLIB_IMPORT = b"\nimport matplotlib.pyplot as plt\n"
_FIRST_PLOT_CALL = b"\n  plt.figure(figsize=(10, 10))\n"
_LAZY_MATPLOTLIB_IMPORT = b"\n  import matplotlib.pyplot as plt"


class PatchError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class RepackResult:
    output_wheel_path: Path
    library_sha256_before: str
    library_sha256_after: str
    library_was_already_patched: bool
    drawing_utils_was_already_patched: bool


def patch_library(library: bytes) -> tuple[bytes, bool]:
    telemetry_count = library.count(_TELEMETRY_ENDPOINT)
    blocked_count = library.count(_BLOCKED_ENDPOINT)
    if telemetry_count == 0 and blocked_count == 1:
        return library, True
    if telemetry_count != 1 or blocked_count != 0:
        raise PatchError(
            f"Unexpected library: telemetry endpoint found {telemetry_count} times, blocked endpoint "
            f"{blocked_count} times. Expected exactly one telemetry endpoint. Inspect this MediaPipe version by hand."
        )
    return library.replace(_TELEMETRY_ENDPOINT, _BLOCKED_ENDPOINT), False


def patch_drawing_utils(source: bytes) -> tuple[bytes, bool]:
    top_level_count = source.count(_TOP_LEVEL_MATPLOTLIB_IMPORT)
    lazy_count = source.count(_LAZY_MATPLOTLIB_IMPORT)
    plot_call_count = source.count(_FIRST_PLOT_CALL)
    if top_level_count == 0 and lazy_count == 1:
        return source, True
    if top_level_count != 1 or lazy_count != 0 or plot_call_count != 1:
        raise PatchError(
            f"Unexpected drawing_utils.py: top-level matplotlib import found {top_level_count} times, lazy import "
            f"{lazy_count} times, first plot call {plot_call_count} times. Inspect this MediaPipe version by hand."
        )
    without_top_level_import = source.replace(_TOP_LEVEL_MATPLOTLIB_IMPORT, b"\n")
    return without_top_level_import.replace(_FIRST_PLOT_CALL, _LAZY_MATPLOTLIB_IMPORT + _FIRST_PLOT_CALL), False


def record_hash(content: bytes) -> str:
    digest = hashlib.sha256(content).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def update_record(record: bytes, contents_by_path: dict[str, bytes]) -> bytes:
    updated_lines: list[str] = []
    for line in record.decode("utf-8").splitlines():
        path = line.rsplit(",", 2)[0]
        if path in contents_by_path:
            content = contents_by_path[path]
            line = f"{path},{record_hash(content)},{len(content)}"
        updated_lines.append(line)
    return ("\n".join(updated_lines) + "\n").encode("utf-8")


def find_record_path(wheel: zipfile.ZipFile) -> str:
    record_paths = [name for name in wheel.namelist() if name.endswith(".dist-info/RECORD")]
    if len(record_paths) != 1:
        raise PatchError(f"Expected exactly one RECORD in the wheel, found {len(record_paths)}.")
    return record_paths[0]


def verify_wheel(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        record_path = find_record_path(wheel)
        record_lines = wheel.read(record_path).decode("utf-8").splitlines()
        for line in record_lines:
            path, expected_hash, expected_size = line.rsplit(",", 2)
            if path == record_path:
                continue
            content = wheel.read(path)
            if record_hash(content) != expected_hash or str(len(content)) != expected_size:
                raise PatchError(f"RECORD does not match {path} in {wheel_path}.")
        library = wheel.read(_LIBRARY_PATH)
        if library.count(_TELEMETRY_ENDPOINT) != 0 or library.count(_BLOCKED_ENDPOINT) != 1:
            raise PatchError(f"The library in {wheel_path} is not patched.")
        if wheel.read(_DRAWING_UTILS_PATH).count(_TOP_LEVEL_MATPLOTLIB_IMPORT) != 0:
            raise PatchError(f"drawing_utils.py in {wheel_path} still imports matplotlib at the top.")


def repack_mediapipe_wheel(source_wheel_path: Path, output_folder: Path) -> RepackResult:
    output_folder.mkdir(parents=True, exist_ok=True)
    output_wheel_path = output_folder / source_wheel_path.name
    temporary_path = output_wheel_path.with_name(output_wheel_path.name + ".repacking")
    with zipfile.ZipFile(source_wheel_path) as source_wheel:
        record_path = find_record_path(source_wheel)
        library = source_wheel.read(_LIBRARY_PATH)
        patched_library, library_was_already_patched = patch_library(library)
        patched_drawing_utils, drawing_utils_was_already_patched = patch_drawing_utils(
            source_wheel.read(_DRAWING_UTILS_PATH)
        )
        patched_contents_by_path = {
            _LIBRARY_PATH: patched_library,
            _DRAWING_UTILS_PATH: patched_drawing_utils,
        }
        patched_contents_by_path[record_path] = update_record(
            source_wheel.read(record_path), patched_contents_by_path
        )
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as output_wheel:
            for entry in source_wheel.infolist():
                content = patched_contents_by_path.get(entry.filename)
                if content is None:
                    content = source_wheel.read(entry)
                output_wheel.writestr(entry, content, compress_type=zipfile.ZIP_DEFLATED)
    verify_wheel(temporary_path)
    temporary_path.replace(output_wheel_path)
    return RepackResult(
        output_wheel_path=output_wheel_path,
        library_sha256_before=hashlib.sha256(library).hexdigest().upper(),
        library_sha256_after=hashlib.sha256(patched_library).hexdigest().upper(),
        library_was_already_patched=library_was_already_patched,
        drawing_utils_was_already_patched=drawing_utils_was_already_patched,
    )


def print_result(result: RepackResult) -> None:
    print(f"Wheel:                   {result.output_wheel_path}")
    print(f"Library SHA-256 before:  {result.library_sha256_before}")
    print(f"Library SHA-256 after:   {result.library_sha256_after}")
    print(f"Library:                 {'already patched' if result.library_was_already_patched else 'patched'}")
    print(f"drawing_utils.py:        {'already patched' if result.drawing_utils_was_already_patched else 'patched'}")
    print("RECORD verified.")


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description="Copy a mediapipe wheel with the telemetry endpoint blocked and matplotlib imported lazily."
    )
    argument_parser.add_argument("source_wheel", type=Path)
    argument_parser.add_argument("output_folder", type=Path)
    arguments = argument_parser.parse_args()
    try:
        result = repack_mediapipe_wheel(arguments.source_wheel.resolve(), arguments.output_folder.resolve())
    except PatchError as error:
        print(error, file=sys.stderr)
        return 1
    print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
