# Research

Gathered 2026-11-02. Re-check anything time-sensitive (versions, prices) before relying on it.

## Existing tools

### BlendCap (Arcomade): the closest competitor
Source: https://superhivemarket.com/products/blendcap

- Blender add-on, $60 one-time (team/studio tiers), GPL. v1.0.5, August 2026. Blender 4.2–5.2.
- Single camera; body, hands/fingers, face, world-space travel. Two-camera capture on its roadmap.
- Local processing; internet only for a one-time setup that downloads **~11 GB of models** from their Hugging Face
  mirrors into a private Python environment. Says nothing is uploaded (their claim, not verified by us).
- Retarget presets: Rigify, Auto-Rig Pro, CloudRig, **Mixamo**; visual bone-map editor for custom rigs.
- Cleanup: foot lock/grounding, depth-noise filter, smoothing, camera-angle correction, tracking-failure repair.
- Exports BVH, FBX; saved capture library.
- NVIDIA GPU recommended (6 GB+ VRAM); CPU works but slowly; ~32 GB disk. Windows and Linux, no macOS.
- Does not name its models; the size suggests heavy 3D body models, not MediaPipe.

### FreeMoCap
Sources: https://github.com/freemocap/freemocap, https://www.pypi.org/project/ajc27-freemocap-blender-addon/0.2.13/

- Free, research-grade, AGPL-3.0 (commercial licensing negotiable).
- Mainly multi-camera. Since v1.4.0, 3D from one camera is possible but marked less reliable; 2D is the default.
- Separate application: `pip install freemocap[cuda|cpu]`, Python 3.9–3.11, desktop UI (Electron); tracking via its
  `skellytracker` package.
- Blender through an export add-on (ajc27) that builds a Rigify rig + body mesh; our Mixamo rigs would need another
  retarget step.

### BlendArMocap (cgtinker)
Source: https://www.blendernation.com/2022/06/08/blendarmocap-free-motion-capture-add-on-release-version/

- Free MediaPipe add-on inside Blender, transfers to Rigify. Released 2022; believed unmaintained (not verified).

### Others seen
- VIPER: MediaPipe face capture for Blender (face only).
- PosePipe (klich3): MediaPipe + OpenCV live/webcam capture in Blender.
- Commercial cloud services (DeepMotion, QuickMagic, Rokoko Vision, Move.ai): ruled out, since they need uploads.

### Where FreeClipMocap fits
- Better than BlendCap on: free, small (~100–200 MB vs 11 GB), CPU is enough, code we have read (network behaviour
  verified), and multi-camera later.
- Better than FreeMoCap on: inside Blender, one install, direct to Mixamo rigs.
- Gap nobody fills today: **multi-camera capture inside Blender**.
- Weaker: single-camera quality at first (MediaPipe's depth is rough vs heavy 3D body models).

## Models considered

- **MediaPipe** (Apache 2.0): chosen. Pose Landmarker (33 landmarks, image + metric world coordinates;
  lite/full/heavy variants) and Hand Landmarker (21 landmarks per hand). Runs on CPU.
- **SMPL-based 3D models** (GVHMR, WHAM, 4D-Humans): better single-camera depth and direct joint rotations, but the
  SMPL body model is **non-commercial** (commercial licence via Meshcapade) and they want an NVIDIA GPU. Not chosen.
- **RTMPose / RTMW** (OpenMMLab, official ONNX): body + fingers but 2D only; would need a depth-lifting model.
  Not chosen.

## MediaPipe package (PyPI, checked 2026-11-02)
Source: https://pypi.org/pypi/mediapipe/1.2.0/json

- Latest **1.2.0**, uploaded 2026-10-27. Major jump from the 0.10.x line: verify the API in Phase 0.
- Wheels are `py3-none-<platform>` (not tied to a CPython version): `win_amd64`, `win_arm64`,
  `manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`, `macosx_11_0_arm64`. `requires_python` not set.
- Dependencies: `absl-py~=2.3`, `certifi`, `numpy`, `sounddevice~=0.5`, `flatbuffers~=25.9`,
  `opencv-contrib-python`, `matplotlib`.

## Blender (on the user's PC)

- Blender 5.2.3 LTS: Python **3.13.13**, numpy **2.3.4** (checked by running Blender, 2026-11-02).
- Extensions (Blender 4.2+): `blender_manifest.toml`; Python wheels can be bundled and Blender installs them into the
  extension's folder; network and file access are declared as permissions. Details of extensions.blender.org policy
  (bundled executables, size limits, per-platform zips) are **not yet checked**.
- extensions.blender.org rules (checked 2026-11-03):
  - Licence: add-ons **GPL-3.0-or-later** (https://docs.blender.org/manual/en/dev/extensions/licenses.html).
    MediaPipe and OpenCV (Apache-2.0) are compatible with GPLv3.
  - Add-on guidelines (https://developer.blender.org/docs/handbook/extensions/addon_guidelines/): third-party
    modules only as bundled wheels; no installing packages at runtime; **no inserting modules into the global
    module dictionary or changing the module search path**; no writing into the add-on's own folder (use
    `bpy.utils.extension_path_user`); respect `bpy.app.online_access`.
  - Moderation guidelines (https://developer.blender.org/docs/features/extensions/moderation/guidelines/):
    "self-contained", "should not have functionality which is extended by external software", no auto-updater,
    no dependence on external servers (localhost is fine). Subprocesses are not forbidden. **Bundled native
    executables are not mentioned either way.**
  - Upload cap: 200 MB per file, per a maintainer in
    https://projects.blender.org/infrastructure/extensions-website/issues/130; per-platform zips are the
    workaround for large extensions (issue 222). Not in official docs.
  - Our Windows zip estimate (all Python): wheels ~76 MB + models ~51 MB (lite 5.5, full 9.0, heavy 29.2,
    hand 7.5) ≈ 128 MB.
- Blender add-on code that uses `bpy` must be GPL-compatible; the whole project is GPL-3.0-or-later.

## MediaPipe telemetry (found 2026-11-02, Phase 0)

The PyPI wheels of MediaPipe (0.10.35 and later, including **1.2.0**) report usage to Google. Checked in the
1.2.0 `win_amd64` wheel:

- Since 1.x the Python package is a thin `ctypes` wrapper around one native library, `mediapipe/tasks/c/libmediapipe.dll`
  (64 MB). The Python code itself makes no network calls.
- The DLL imports `WININET.dll` (`InternetOpenA`, `HttpSendRequestA`) and contains Google's internal Clearcut logger
  (`mediapipe/tasks/cc/core/logging/google_internal/clearcut_logging_client.cc`, `portable_clearcut_uploader.cc`,
  `flat_file_log_store.cc`) posting to **`https://play.googleapis.com/log`**, plus MediaPipe analytics protos
  (`mediapipe/util/analytics/...`).
- No environment variable, flag or option to turn it off was found in the DLL or the Python API.
- It also contains Hugging Face download URLs for speech/TTS models (`huggingface.co/litert-community/...`), used by
  audio/GenAI tasks, not by the vision landmarkers when a model path is given.
- Upstream issue https://github.com/google-ai-edge/mediapipe/issues/6291 (closed 2026-07-06). Maintainer: no official
  API to disable it; "build the SDK from source, which will not include telemetry, or block access to the host";
  input data is said never to be sent (not independently verified). The open-source repo has no Clearcut code: the
  PyPI wheels are built inside Google.
- 0.10.21 was reported clean; others pinned `mediapipe<=0.10.21` to avoid it.
- What it reports (from the Python `BaseOptions`): host environment (Python), OS, Python version, plus a CA bundle
  path for HTTPS (why `certifi` is a dependency). Event contents beyond that are not known.
- **Patch:** the wheel repack tool (`Tools/repack_mediapipe_wheel.py`; first written in C# as
  `PatchMediaPipeTelemetry.cs`, retired 2026-11-03) replaces the single occurrence of `https://play.googleapis.com/log`
  with the same-length `https://127.0.0.1:1/blocked0000` (local, refused at once, no DNS lookup). 1.2.0 `win_amd64`
  DLL SHA-256: original `C979DF9F…11D8`, patched `A521BBA6…B8D4`. The only other `googleapis` strings are protobuf
  type names (`type.googleapis.com/...`), not network addresses.
- **Verified 2026-11-02** with the patched DLL in headless Blender (3 pose tasks + 1 hand task, 90 s idle): exactly
  4 connection attempts, all to `127.0.0.1:1`, made **60 s after the tasks were created** (one per task); no other
  connections, no Google/Hugging Face entries in the DNS cache.
- Importing MediaPipe imports matplotlib (`drawing_utils`), which writes a font cache to `%LOCALAPPDATA%\matplotlib`.
  Avoided by the lazy-import patch in `drawing_utils.py` (see "Dependencies" below).
- **The logger also flushes when a task is closed**, not only after 60 s: in the offline chain dry run
  (2026-11-02), each of the two tasks made one attempt to `127.0.0.1:1` at the moment it was closed (~5 s into
  the run). Both refused locally; the patch covers this path too.

## Dependencies (checked 2026-11-02, Phase 0)

Checked with `Scratch/Phase0/scripts/check_dependencies.py` (hides packages from the import system, imports
MediaPipe in headless Blender, runs pose + hands) and then for real with `site-minimal/` (only the two kept
packages on disk) in `offline_chain_test.py`.

| Package (+ its dependencies) | Wheels | Used by | Result |
|---|---|---|---|
| sounddevice (+ cffi, pycparser) | 1.2 MB | Audio tasks only; import already in `try/except ImportError` | Dropped |
| matplotlib (+ contourpy, cycler, fonttools, kiwisolver, packaging, pillow, pyparsing, python-dateutil, tzdata) | 19.6 MB (~75 MB unpacked) | `drawing_utils.plot_landmarks` (debug 3D plot); but `vision/__init__.py` imports it unconditionally | Dropped; bundled `drawing_utils.py` patched to import it inside `plot_landmarks` only (2026-11-03). An earlier `sys.modules` stand-in broke the extension guidelines |
| absl-py, flatbuffers | 0.16 MB | CLI tool, tests, metadata writers; never loaded on our path | Dropped |
| certifi | — | `base_options.py` (CA bundle for the telemetry upload) | Blender 5.2 ships its own (2025.04.26) |
| numpy | — | everything | Blender's 2.3.4 |
| opencv-contrib-python | 52.6 MB | MediaPipe's `drawing_utils`; our video decoding | Kept |
| mediapipe | 23.9 MB | — | Kept |

Bundled wheels: ~76 MB instead of ~98 MB; unpacked 207 MB instead of 290 MB.

## C# subprocess link (checked 2026-11-02, Phase 0; superseded 2026-11-03 by the all-Python decision)

`Scratch/Phase0/CoreProbe` (.NET 10, framework-dependent build, runtime 10.0.12) run from headless Blender
5.2 by `scripts/run_core_probe.py`:

- Process start to first stdout line: ~80 ms warm, ~190 ms cold.
- stdout lines arrive as written (.NET's `Console.Out` auto-flushes), so progress can be shown live.
- Paths and text with spaces and non-ASCII characters (`Prüfung ✓`) round-trip both ways once the exe sets
  `Console.OutputEncoding` to UTF-8 without a byte-order mark and Python reads the pipes as UTF-8.
- Errors on stderr with exit code 1; `Popen.terminate()` stops it at once.
- `subprocess.CREATE_NO_WINDOW` keeps a console window from flashing up when Blender runs with its UI.
- stdout and stderr are each drained on a reader thread, so a full pipe can never block the exe; Blender's main
  thread only polls a queue (the pattern the add-on will use with a timer or modal operator).
- A framework-dependent build needs the .NET 10 runtime on the user's machine, hence Native AOT in Phase 3.

## First real clip (2026-11-02, Phase 0)

- The user's own clip, run by the user alone: body tracking looked good with all three pose models.
- Fingers were sometimes lost with relaxed hands edge-on to the camera (fingers at 90° to the view). Causes: the Hand
  Landmarker starts with a palm detector, which fails without a visible palm, and hands in a full-body shot are only
  ~50–100 px tall. Remedies planned for Phase 2: hand crops around the pose wrists, gap filling, fallback pose
  (decided 2026-11-03: hold the last good pose); filming at 4K or closer helps.

## First capture with the extension (2026-11-03, Phase 1)

The user's clip (1280×720, 30 fps, 327 frames, 10.9 s), heavy model, captured in Blender's UI on the background
thread:

- Body found in 326 of 327 frames (missing only in frame 0). Left hand 195 frames (60%), right hand 169 (52%);
  both 89, neither 52. Longest gaps 33 (left) and 47 (right) frames, so hand gap filling and a fallback pose matter.
- Matching hands to pose wrists works: hand wrist to its own pose wrist median 5–6 px (max 35 px), never closer
  to the other wrist, so no left/right swaps.
- World landmark bone lengths vary by 1–2.6 cm (std) through the clip, and left/right differ a lot (upper arm
  20 vs 28 cm, thigh 38 vs 31 cm): single-camera depth error. The solver should use bone **directions** against the
  rig's own lengths, and expect directions toward/away from the camera to be the weakest.
