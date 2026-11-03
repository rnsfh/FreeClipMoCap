# Roadmap

Estimates made 2026-11-02 (for the C# plan; all Python since 2026-11-03 removes the subprocess link and the Native
AOT build, so Phases 1 and 3 may come out a little shorter). A "session" is a solid working block with Claude Code
(writing, running Blender headless, checking renders, fixing). A session typically processes 2–5 M tokens (mostly
cached context) and outputs 50–150 k tokens. **Recalibrate after Phase 0** by comparing `/usage` before and after it.

The bottleneck is testing with real footage (the user films, both review), not writing code.

| Phase | Content | Sessions | Calendar (1–2 sessions most days) |
|---|---|---|---|
| 0. Feasibility | MediaPipe inside Blender's Python, dependencies, offline test | 1 | 1 day |
| 1. MVP | Video → body animation on a Mixamo rig, bone map, smoothing, FBX/BVH export | 4–8 | 1–2 weeks with Phase 0 |
| 2. Quality | Foot lock, in place, fingers, compare overlay, failure repair, more rig presets, tuning on real clips | 5–10 | 3–5 weeks for Phases 0–3 |
| 3. Packaging | Manifest, patched + bundled wheels, docs, Windows build, submission | 2–4 | (included above) |
| 4. Multi-camera | Calibration board, video sync, triangulation, UI | 6–12 | another 3–6 weeks |

Totals: single camera polished (0–3) ≈ 12–23 sessions ≈ 30–100 M tokens processed; with multi-camera
≈ 18–35 sessions ≈ 50–170 M tokens.

## Phase 0: feasibility (everything in a scratch folder, Blender install untouched)

- [x] Download MediaPipe (latest) and its dependency wheels with Blender's `python.exe -m pip download`
      (download only). Name every URL to the user first.
- [x] Download the Pose Landmarker and Hand Landmarker `.task` models (lite, full, heavy + hands).
- [x] Read MediaPipe's code for network calls or telemetry; report findings. **Telemetry found** (RESEARCH.md).
- [x] Neutralise the telemetry: `PatchMediaPipeTelemetry.cs` (C#, since ported into the Python repack tool);
      verified with a connection watch.
- [x] Check MediaPipe 1.x's API: `mediapipe.tasks.python.vision.PoseLandmarker` / `HandLandmarker`, VIDEO mode works.
- [x] Install into a throwaway target folder and import it from Blender 5.2's Python 3.13.
- [x] Run pose + hands on a real test video (the user runs `Scratch/Phase0/scripts/compare_pose_models.py`
      alone; see `Scratch/Phase0/README.md`); compare lite / full / heavy. All three looked good to the user
      (2026-11-02); fingers sometimes lost when the hands were relaxed and edge-on to the camera.
- [x] Check which dependencies are really needed. Only `mediapipe` + `opencv-contrib-python` are bundled;
      matplotlib is avoided by a lazy-import patch in `drawing_utils.py` (RESEARCH.md, "Dependencies").
- [x] Build a minimal C# console app and run it from Blender via `subprocess` (`Scratch/Phase0/CoreProbe`,
      `scripts/run_core_probe.py`): worked, then **superseded** by the all-Python decision (2026-11-03).
- [x] Offline test with the internet disconnected (the user runs `Scratch/Phase0/scripts/offline_test.ps1`):
      generated video → MediaPipe (minimal dependencies) → C# probe. **Passed 2026-11-02**: chain works, only
      connections were the 2 patched telemetry attempts to `127.0.0.1:1` (refused locally). Re-run online with
      the lazy-import patch (2026-11-03): works, matplotlib never loaded.
- [ ] Measure token usage of the session.

## Phase 1: MVP

- [x] Settle the source layout, the landmark cache layout, and the first Python style choices (DECISIONS.md,
      2026-11-03).
- [x] Wheel repack tool (Python): both mediapipe patches, `RECORD` update; fills `src/freeclipmocap/wheels/` and
      `models/`. Development install by a folder junction (2026-11-03; output byte-identical to the Phase 0 patch).
- [x] Extension skeleton: manifest (GPL-3.0-or-later), panel, operators, bundled wheels (2026-11-03).
- [x] Capture operator with progress bar, landmark cache; Blender's UI stays responsive (2026-11-03, background
      thread; smooth in the UI on the user's clip).
- [x] Bone map: auto-guess from names and hierarchy, editable map in the panel, Mixamo preset. Done 2026-11-03,
      with save/load of JSON presets; all 12 LifeInArena heroes map fully (52 roles) with no edits.
- [x] Solver (numpy): landmark reader, swing/twist solver for body bones against the armature's rest pose,
      smoothing. Done 2026-11-03, plus visibility gating, left/right glitch skipping, camera leveling; checked on the
      user's clip with the human hero (numbers only).
- [x] Key reduction (default on, 2° tolerance; 2026-11-03).
- [x] Write the Action; FBX and BVH export (2026-11-03). FBX take named after the take and starting at 0 s; BVH by
      our own Y-up writer. Checked in Godot 4.7 with LifeInArena's Mixamo import settings against idle_female: same
      size, upright, facing the camera. The bone map's node path is `Armature/Skeleton3D` (Mixamo: `Skeleton3D`).
- [x] Show Video (video beside the viewport, synced); Solve matches the scene frame rate to the video (2026-11-03).
- [x] First real clip filmed by the user; reviewed on a LifeInArena hero in Godot (2026-11-03). The pipeline works
      end to end, but the user is not happy with the motion quality yet and puts it down to the single camera and
      the size of the AI model; both to address later (Phase 2 tuning, Phase 4 multi-camera, possibly another model
      through the tracking table).
- [x] Start/end frame of the video, used by Solve (2026-11-03).

## Phase 2: quality

- [x] Foot contact detection and locking (leg IK); floor at the rig's rest foot height (2026-11-03). On the user's
      clip with the human hero: fully locked stretches hold the ankle to 0.0 mm sideways, the lowest foot point
      stays on the floor. Contacts assume a static camera.
- [x] Root motion vs in place: "Root Motion" checkbox, off by default (2026-11-03).
- [ ] Fingers from the Hand Landmarker; fallback pose when a hand is lost.
- [x] Hand crops around each pose wrist, one tracker per hand (2026-11-03): hands found 88% / 49% on the user's latest
      clip (was 1% / 12%). Gaps blended over time. Open: MediaPipe's finger points are very jittery on hands ~30 px
      from wrist to knuckle (720p, full body in frame). The user chose a filming note over filters
      ([FILMING.md](FILMING.md): 1080p/4K, hands at least ~120 px); to re-check on a sharper clip.
- [x] Compare view: video behind the rig through a camera fitted to the clip, Lens setting (2026-11-03; beside the
      rig with the tracked skeleton drawn on it: done in Phase 1). On the user's clip with the human hero: joints a
      median 29 px (4% of the height) from the tracked points; to look at in Blender's UI.
- [x] Tracking-failure repair: finger poses no longer snap when a hand is found again (gap blending); impossible
      finger bends clamped by joint limits (2026-11-03).
- [ ] ~~Ground plane from the feet instead of leveling by the torso~~ moved to Phase 4 (the user, 2026-11-03): on the
      user's clip a floor fitted through planted heels and toes disagreed with the torso by 21° (single-camera foot
      depth is too rough). Torso leveling stays; an actor leaning the whole clip is still straightened.
- [x] Bone-map presets: built-in Rigify on its FK controls, with Solve switching the rig to FK and the writer
      correcting for the rig's constraints (2026-11-03). Auto-Rig Pro, Unreal mannequin, VRoid/VRM: not chosen for now.
- [x] Key reduction keeps planted feet within ~5 mm (tighter tolerance for hips and legs, 2026-11-03).
- [ ] Tuning on several real clips (idle, taunt, two-hand swing, hit reaction, death).

## Phase 3: packaging

- [x] Re-check extensions.blender.org rules and the upload cap (2026-11-03): the add-on guidelines are unchanged
      (no network unless allowed, no runtime installs, bundled wheels, no writes to the add-on folder); the manifest
      passes Blender's validator, with `copyright` added. No official upload cap is published; a 213 MB zip was
      refused (the site suggests one zip per platform); ours is 126 MB.
- [x] User docs: filming guide ([FILMING.md](FILMING.md)), install, workflow, panel and troubleshooting at the top of
      the README (2026-11-03).
- [x] Check that Blender's extension installer installs only the listed wheels (2026-11-03): a zip install into a
      throwaway user folder gives only `mediapipe` and `cv2`; MediaPipe's declared dependencies are not pulled in.
      `Tools/build_extension.py` repeats this check and runs all tests on the installed copy at every build.
- [ ] Re-verify the repacked mediapipe wheel (both patches; the tool is written in Phase 1) on every MediaPipe
      update; reconsider building libmediapipe from source instead.
- [x] Windows build: `Tools/build_extension.py` → `dist/freeclipmocap-0.1.0.zip` (2026-11-03). Linux/macOS later
      (other wheels; MediaPipe's telemetry patch must be redone for each platform's library).
- [ ] Submit to extensions.blender.org.

## Phase 4: multi-camera

- [ ] Calibration with a printed board (ChArUco; OpenCV's aruco module is in `opencv-contrib-python`).
- [ ] Video sync (clap in the audio, or a light flash).
- [ ] Triangulation of 2D landmarks from several views; per-view confidence weighting.
- [ ] UI for several videos.
- [ ] Floor from the calibration board or the triangulated feet (moved from Phase 2).
