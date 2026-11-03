# Decisions

## Made (by the user)

| Decision | Choice |
|---|---|
| Approach | Own tool, not a paid or cloud service |
| Pose model | MediaPipe, **latest version** (1.2.0 at the time) |
| Where it runs | Inside Blender, as a Blender extension |
| Language (2026-11-03) | **All Python** (numpy for the math), so the extension is only Python + bundled wheels as extensions.blender.org expects. Replaces the first plan (thin Python add-on + C# core run as a subprocess, 2026-11-02). The build tools are Python too (2026-11-03) |
| Cameras | Single camera first; multi-camera later if it goes well |
| Privacy | Fully local, no third parties (see [PRIVACY.md](PRIVACY.md)) |
| Distribution | extensions.blender.org, installed from inside Blender in one step (2026-11-03) |
| Name (2026-11-03) | **FreeClipMocap**. The first name, BlendMotion, was taken. Checked free 2026-11-03: no extension on extensions.blender.org, no GitHub repo, no web hits. Close to the existing FreeMoCap project (noted to the user). The repo folder stays `BlendMotion` unless the user renames it |
| Licence (2026-11-03) | **GPL-3.0-or-later** for the whole project (required for add-ons on extensions.blender.org) |
| Start | Workspace and docs only; implementation (Phase 0) later |
| Wheel downloads (Phase 0) | With Blender's own `python.exe`, not the standalone Python 3.15 |
| Scratch work | `Scratch/` in the repo (wheels, models, experiments) |
| Pose model variant | **Heavy** by default; lite and full offered as a fast preview (decided after the first real clip) |
| Re-solving | Every Solve makes a new Action (take): `Name`, `Name.001`, …, each with a fake user so Blender's cleanup on save keeps it; the newest is assigned to the armature |
| Landmark cache | One cache per video + model + capture settings (named with a short settings hash), in a FreeClipMocap folder next to the video; identical settings reuse it |
| Landmark cache format (2026-11-03) | **Binary**; the exact layout is proposed before Phase 1 code. (Bone rotations no longer need a file: the solver runs in the same Python as the keyframing.) |
| FBX export over an existing file | Blender's normal file dialog: red name, the user confirms the overwrite |
| MediaPipe telemetry | Keep 1.2.0 and patch the endpoint in `libmediapipe.dll` (first with `PatchMediaPipeTelemetry.cs`, now inside the Python wheel repack tool); building from source is revisited in Phase 3 |
| Bundled wheels | Only `mediapipe` and `opencv-contrib-python`. Dropped: sounddevice, cffi, pycparser, absl-py, flatbuffers, and matplotlib with its 9 dependencies. certifi and numpy come from Blender |
| matplotlib (2026-11-03) | Not bundled. The bundled mediapipe wheel gets a second patch: `drawing_utils.py` imports matplotlib inside `plot_landmarks` instead of at the top. (The earlier `sys.modules` stand-in broke the add-on guideline "Bundle Modules".) |
| Offline test | The user disconnects the internet and runs `Scratch/Phase0/scripts/offline_test.ps1` alone (not firewall rules). Passed 2026-11-02 |
| Video decoding (2026-11-03) | OpenCV for capture; Blender's movie clip only for showing the video in the Compare view |
| Target rigs (2026-11-03) | Versatile, best user experience: auto-guess the bone map from names and hierarchy → editable map in the panel → save/load maps as presets; built-in presets for Mixamo (Phase 1), Rigify, Auto-Rig Pro, … (Phase 2) |
| Export (2026-11-03) | From the panel: FBX (LifeInArena settings, Blender's built-in exporter) and BVH (our own writer, see "BVH export") |
| Python style (2026-11-03) | Same spirit as the C# rules; whenever a Python implementation choice comes up, ask the user |
| Multi-camera calibration (2026-11-03) | Python (OpenCV), the user's leaning; confirm in Phase 4 |
| Source layout (2026-11-03) | `src/freeclipmocap/` is the extension root, with subpackages `ui/`, `rig/`, `capture/`, `solver/`, `animation/`, `constants/`; `models/` and `wheels/` are filled by the build tool. Names lowercase snake_case (PEP 8). `src/Core` and `src/Addon` removed |
| Solver free of `bpy` (2026-11-03) | Yes: `solver/` imports only numpy and `capture/` only mediapipe + cv2, so both are tested headless with Blender's own `python.exe` |
| Landmark cache layout (2026-11-03) | numpy `.npz` (uncompressed), named arrays plus a format version and the capture settings; always loaded with `allow_pickle=False`. Exact arrays left to Claude by the user (2026-11-03); see ARCHITECTURE.md, "Landmark cache" |
| Hand tracking (2026-11-03) | **Always on**, no option in the panel (the user) |
| Capture thread (2026-11-03) | **Background thread** inside Blender (MediaPipe's ctypes calls release Python's GIL). Only the thread touches cv2/mediapipe; only the main thread touches `bpy`. mediapipe and cv2 are imported when a capture starts, not when the extension loads. Confirmed by the user on a real clip: Blender stayed smooth |
| Fingers when a hand is lost (2026-11-03) | ~~Hold the last good pose~~ replaced the same day by **blend across gaps** (the user): gaps up to 1 s ease from the last good hand pose to the next; longer gaps hold, then ease into the next good pose over 0.25 s. Before the first good frame, the first good pose is held |
| Hand search (2026-11-03) | **A crop around each wrist, one hand tracker per hand** (the user), instead of searching the full frame (hands were found in 1–12% of the user's latest clip). Tuned by Claude on that clip: the crop is 2.2 torso lengths wide (15 hand lengths when the torso is foreshortened, at least 96 px), centred between wrist and knuckles, never scaled down (MediaPipe's palm finder needs the context; its finger model needs the pixels); a hand counts when its wrist lies within 12% of the crop from the body's wrist. Hands found: 88% / 49%. Captures redone once (settings field `hand_search`) |
| Finger joint limits (2026-11-03) | **Yes** (the user): each finger joint's rotation is split into curl (toward the palm), spread and twist, each clamped: base joints curl −30…95°, spread ±20°; middle −5…110°; end −10…90°; middle/end spread ±5°; thumb looser. Palm side from the thumb tip; a hand without a mapped thumb is not limited |
| Bone roles (2026-11-03) | **Mixamo names** (the user): Hips, Spine, Spine1, Spine2, Neck, Head, Left/RightShoulder, Arm, ForeArm, Hand, Hand{Thumb,Index,Middle,Ring,Pinky}{1,2,3}, UpLeg, Leg, Foot, ToeBase. Required: Hips, Spine, Head, arms (Arm, ForeArm, Hand), legs (UpLeg, Leg, Foot); the rest optional. Tip bones (`…4`, `_End`) are never keyed; unmapped bones (tails, `headfront`) are left alone |
| Bone map storage (2026-11-03) | On the **armature object** (property group), saved in the .blend with it |
| Bone map presets (2026-11-03) | **JSON** `{role: bone name}`; the user's in the extension's user folder (`extension_path_user`), built-in ones shipped read-only inside the extension: Mixamo, and **Rigify** (the user, Phase 2; the only rig testable here: generated headless in the tests). Unreal mannequin, VRoid/VRM and Auto-Rig Pro not chosen for now |
| Control rigs (2026-11-03) | **Solve switches the rig to FK** (the user): a preset may carry `rig_settings` (the rig's own sliders) that Solve applies and reports; Rigify's arms and legs go to FK (`IK_FK` = 1, measured: 1 = FK) and neck/head follow to 0. The action writer then corrects bones moved by the rig's constraints until Blender's evaluated pose matches the solve (Claude, needed for Rigify: without it 38 of 52 controls were off by up to 55°) |
| Key reduction near planted feet (2026-11-03) | **Tighter tolerance for the hips and legs** (the user): 2° per bone let planted feet slide 6–8 cm; bones above a foot now get the tolerance that moves the foot by 0.2% of the hips-to-toe length. On the user's clip: feet within ~5 mm, legs keep ~69% of keys (jittery tracking), arms ~29% |
| Copyright (2026-11-03) | `copyright = ["2026 rnsfh"]` in the manifest (the user) |
| Building the zip (2026-11-03) | `Tools/build_extension.py` (the user): output in `dist/` (deleted and recreated), Blender's own builder, zip checks, then an install into a throwaway Blender user folder with all tests run against it |
| User docs (2026-11-03) | **At the top of the README** (the user): install, a typical session, the panel, limits, troubleshooting; filming in `Docs/FILMING.md`. Developer content follows under "For developers" |
| Floor from the feet (2026-11-03) | **Moved to Phase 4** (the user): single-camera foot depth too rough (21° off the torso on the user's clip) |
| Preset matching (2026-11-03) | A preset counts in the auto-guess only when it covers every required role (Claude; Rigify's `head`/`neck` otherwise "matched" unrelated rigs) |
| Bone map auto-guess (2026-11-03) | **Presets first** (best match, ignoring a namespace prefix such as `mixamorig:`), **then names + hierarchy** for roles still empty. Result shown in the editable map |
| Solver clean-up rules (2026-11-03) | Chosen by Claude under "do what's best for users", after the user's clip showed the need: visibility < 0.5 = missing; a point visible in < 25% of the clip is not used (its bones stay unanimated, reported); single-frame left/right mix-ups skipped; camera leveled by the median torso direction (reported in degrees); head ear→eye line taken as 10° above level in a neutral head. All in the MediaPipe table or solver constants, easy to tune |
| Key reduction default (2026-11-03) | **2° tolerance** (Claude, under "do what's best"; replaces the planned ~0.5°): on the user's clip 0.5° kept 164 of 326 keys per curve, 2° keeps 84, and 2° is under 2 cm at the hand of a 50 cm arm, far below the capture's own error. Checked against Blender's own curve evaluation, so every frame stays within the tolerance |
| FBX scale settings (2026-11-03) | Blender exporter with `global_scale=0.01`, `apply_unit_scale=True`, `FBX_SCALE_ALL`: the file declares centimetres (`UnitScaleFactor` 1), the armature node carries no ×100, and bone positions are metre values (Hips ≈ 1.09), which is exactly how Mixamo's files are built. Difference left: Blender always writes an `Armature` parent node (−90° on X, cancelled by +90° on the Hips); Mixamo files have none. To confirm in Godot |
| Compare view, first version (2026-11-03) | **Show Video** button: the video in a Movie Clip Editor next to the 3D viewport, synced to the take (the user). Video behind the rig through a camera stays for later |
| Scene frame rate (2026-11-03) | **Solve sets the scene to the video's frame rate**, rounded to the nearest standard rate (e.g. 30.08 → 30), and says so (the user) |
| FBX export (2026-11-03) | **Like a Mixamo clip** (centimetre scale, so LifeInArena's Mixamo import settings apply unchanged) and **armature + current take only** (no mesh, no other takes, no added leaf bones) (the user) |
| FBX take name and start (2026-11-03) | **The FBX take is named after the take and starts at 0 s**, like a Mixamo clip (the user). Blender would name it after the scene ("Scene") and start at frame 1's time (1/30 s), a one-frame hitch on looping. The export briefly shifts the take's keys and the scene range to frame 0 and renames the scene, and restores all three afterwards |
| Root motion (2026-11-03) | **"Root Motion" checkbox in Solve, off by default** (the user): off = in place like Mixamo's "In Place" clips (the hips still move up/down, lean and sway; only travel by stepping is taken out); on = the body travels as in the video. LifeInArena removes travel itself anyway |
| Foot locking (2026-11-03) | **Planted feet locked with two-bone leg IK** (the user): no foot sliding while planted; knees adjusted to reach, blended in/out over 3 frames. Details chosen by Claude: planted = heel and toe nearly still in the image (static camera) and not lifted above the other foot; floor = the rig's own rest-pose foot height, up snapped to the nearest armature axis; the hips are lowered only when a straight leg cannot reach its planted foot |
| Video frame range (2026-11-03) | **Solve only** (the user): capture tracks the whole video once; Solve uses the chosen frames. Picked with **Start/End fields plus Set buttons** that take the current frame (the user). Video frame N sits at scene frame N, so the take lands under the matching video and Solve sets the scene range to it (Claude, needed for the Set buttons to work on any frame) |
| BVH export (2026-11-03) | **Our own writer** (`animation/bvh_writer.py`, numpy; the user): **Y-up** (Blender's exporter writes Z-up, which other programs open lying on its back), position channels on the root only, Z-X-Y rotation channels, metres, the scene range. Tested by importing the file with Blender's default (Y-up) settings: same pose within 0.01 mm |
| Other tracking models (2026-11-03) | **Prepare now, table-driven** (the user): the solver reads a per-model table (role → the two points giving its direction, the points giving its twist), not hard-coded MediaPipe indices; the cache records which model and point set produced it. MediaPipe stays the only model. Adding one later = a capture module + a table + new optional roles (e.g. Jaw). Ears/tails cannot come from filming a human; face expressions would more likely drive shape keys than bones |
| Skipped joints (2026-11-03) | Not adopted: no rule yet for a mapped bone aiming past an empty role (the user declined writing it into the design) |
| Purpose of the result (2026-11-03) | The animation is a **starting point that the animator corrects by hand** (the user). Keyframes must be easy to edit |
| Solver output choices (2026-11-03) | Left to Claude by the user ("do what's best for users; I'm not an animator"). **Rotation mode:** each bone's own (quaternion keys for quaternion bones, Euler with continuity for Euler bones); the rig is never changed. **Keys:** one per scene frame first; key reduction (on by default, ~0.5° tolerance) added after the first visual review. **Timing:** resampled to the scene's frame rate, keys on whole frames from the scene start; the scene end is set to the take's end. **Smoothing:** One Euro filter on the landmarks, one 0–1 slider, default 0.5. **Gaps:** hold the last good pose; before the first good frame, the first good pose. **Hips:** rotation only in Phase 1 (in place); root motion in Phase 2 |
| Data objects (2026-11-03) | `@dataclass(frozen=True, slots=True)`, the counterpart of a sealed record; bulk per-frame data stays in numpy arrays inside them |
| Creating objects (2026-11-03) | Simple data objects are constructed directly; when building does real work, a named `@classmethod` does it (`BoneMap.guess_from_armature(...)`, `LandmarkCache.load(...)`). Changes of state through named methods |
| Constants (2026-11-03) | `freeclipmocap/constants/`, one module per topic, `UPPER_SNAKE_CASE`; a value used only inside one module is a `_LEADING_UNDERSCORE` constant at its top |
| Type hints (2026-11-03) | On every parameter, return value and field; checked by Pylance in basic mode. bpy objects typed loosely (no official stubs) |
| Tests (2026-11-03) | `unittest` (built in), run with Blender's `python.exe` or headless `blender.exe` for bpy parts; `Tests/` mirrors the package layout |
| Build tools (2026-11-03) | Python, run with Blender's `python.exe`. The telemetry patch is ported from `PatchMediaPipeTelemetry.cs` into the wheel repack tool and the C# file retired, so building needs no .NET |
| Patched mediapipe wheel (2026-11-03) | Keeps PyPI's name and version (`mediapipe-1.2.0-py3-none-win_amd64.whl`); only the two patched files and their `RECORD` hashes change. The build tool refuses an unpatched DLL |
| Development install (2026-11-03) | A directory junction from Blender's user extensions folder to `src/freeclipmocap/`, made once by a script; the build tool puts the patched wheels and the models into `src/freeclipmocap/wheels/` and `models/` (generated) |

## Open questions

Ask the user before settling any of these.

1. **Repository:** git init, remote hosting (GitHub?), public or private. Later.
2. **Further Python style specifics**, asked as they come up (data objects, factories, constants, type hints and
   tests are settled above).

## Settled questions (history)

- Exchange formats → binary landmark cache (2026-11-03).
- Licence → GPL-3.0-or-later (2026-11-03).
- Bundled C# executable on extensions.blender.org → avoided by going all Python (2026-11-03). The published
  guidelines neither allowed nor forbade it ("self-contained", "should not have functionality which is extended
  by external software").
- Native AOT prerequisites (Visual Studio C++ workload) → no longer needed (all Python).
- Video decoding → OpenCV. Dependencies → mediapipe + opencv only, matplotlib via lazy import patch.
- Pose model variant → heavy default. Target rigs → auto-guess + editable map + presets. Export → FBX and BVH.
- Name → FreeClipMocap. Other candidates checked free on 2026-11-03: Clip2Rig, Mimicrig, Posewright.
- Source layout, bpy-free solver, `.npz` cache, data classes, factories, constants, type hints, tests, build
  tools in Python, wheel name, development install → see the table (2026-11-03).
