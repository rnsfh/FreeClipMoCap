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
_MODIFICATIONS_NOTICE_NAME = "licenses/MODIFICATIONS"
_MODIFICATIONS_NOTICE = f"""\
This copy of the mediapipe wheel was modified by FreeClipMocap (https://github.com/rnsfh/FreeClipMoCap), as
section 4(b) of the Apache License 2.0 asks. Every other file is as published on PyPI.

Changed files:

{_LIBRARY_PATH}
  The usage-report (telemetry) address {_TELEMETRY_ENDPOINT.decode("ascii")} is replaced by
  {_BLOCKED_ENDPOINT.decode("ascii")}, an address of the same length that cannot be reached, so the library
  cannot send usage reports. Nothing else in the library is changed.

{_DRAWING_UTILS_PATH}
  matplotlib is imported inside plot_landmarks instead of at the top of the module, so importing MediaPipe
  does not need matplotlib.

The hashes in RECORD are updated to match, and this notice is added. The changes are made by
Tools/repack_mediapipe_wheel.py in the FreeClipMocap repository. MediaPipe's own LICENSE and NOTICE are unchanged.
""".encode("utf-8")


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


def record_line(path: str, content: bytes) -> str:
    return f"{path},{record_hash(content)},{len(content)}"


def update_record(record: bytes, record_path: str, contents_by_path: dict[str, bytes]) -> bytes:
    updated_lines: list[str] = []
    listed_paths: set[str] = set()
    for line in record.decode("utf-8").splitlines():
        path = line.rsplit(",", 2)[0]
        listed_paths.add(path)
        if path == record_path:
            continue
        if path in contents_by_path:
            line = record_line(path, contents_by_path[path])
        updated_lines.append(line)
    for path, content in contents_by_path.items():
        if path not in listed_paths:
            updated_lines.append(record_line(path, content))
    updated_lines.append(f"{record_path},,")
    return ("\n".join(updated_lines) + "\n").encode("utf-8")


def modifications_notice_path(record_path: str) -> str:
    return f"{record_path.rsplit('/', 1)[0]}/{_MODIFICATIONS_NOTICE_NAME}"


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
        notice_path = modifications_notice_path(record_path)
        if notice_path not in wheel.namelist() or wheel.read(notice_path) != _MODIFICATIONS_NOTICE:
            raise PatchError(f"{wheel_path} lacks the current modification notice {notice_path}.")


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
        notice_path = modifications_notice_path(record_path)
        patched_contents_by_path = {
            _LIBRARY_PATH: patched_library,
            _DRAWING_UTILS_PATH: patched_drawing_utils,
            notice_path: _MODIFICATIONS_NOTICE,
        }
        patched_contents_by_path[record_path] = update_record(
            source_wheel.read(record_path), record_path, patched_contents_by_path
        )
        source_paths = set(source_wheel.namelist())
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as output_wheel:
            for entry in source_wheel.infolist():
                if entry.filename == record_path and notice_path not in source_paths:
                    notice_entry = zipfile.ZipInfo(notice_path, date_time=entry.date_time)
                    output_wheel.writestr(notice_entry, _MODIFICATIONS_NOTICE, compress_type=zipfile.ZIP_DEFLATED)
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
    print(f"Notice:                  {_MODIFICATIONS_NOTICE_NAME} in the dist-info folder")
    print("RECORD verified.")


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description="Copy a mediapipe wheel with the telemetry endpoint blocked, matplotlib imported lazily and a "
        "notice of these changes added to its dist-info folder."
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
