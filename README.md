# FreeClipMocap

Markerless motion capture inside Blender: film yourself with a phone, get an animation on your rigged character.
Free and open source (GPL-3.0-or-later). Everything runs on your computer: your videos, tracking data and
animations never leave it. The one exception: the bundled MediaPipe tracker, unchanged as Google publishes it,
sends Google usage reports while it tracks (see [Docs/PRIVACY.md](Docs/PRIVACY.md)).

**Status:** Phase 1 (MVP) done; Phase 2 (quality) mostly done (feet, root motion, hand crops, finger limits, the
Rigify preset); Phase 3 (packaging) in progress. See [Docs/ROADMAP.md](Docs/ROADMAP.md).

## Using FreeClipMocap

### Install

Needs Blender 5.2 or newer on Windows (64-bit).

1. Get `freeclipmocap-<version>.zip` (later from extensions.blender.org; for now built with
   `Tools/build_extension.py`, see "For developers").
2. In Blender: **Edit → Preferences → Get Extensions**, the **⌄** menu at the top right → **Install from Disk…**,
   pick the zip. FreeClipMocap is enabled right away.
3. In the 3D viewport press **N** and open the **FreeClipMocap** tab.

The zip is about 150 MB: it brings its own copies of MediaPipe (Google's body and hand tracker), OpenCV and
matplotlib (which MediaPipe needs), and the
tracking models, so nothing is downloaded later.

### Film

One person, whole body in the picture, camera on a tripod. [Docs/FILMING.md](Docs/FILMING.md) has the details
(resolution, hand size for fingers, light).

### A typical session

1. **Video:** pick the video file. Press **Show Video** to see it next to the 3D viewport, synced to the timeline.
2. **Target:** pick your character's armature. Its bone map is guessed (Mixamo and Rigify rigs map completely by
   themselves). Check the line under it: required bones that are missing show in red.
3. **Capture:** press **Capture**. Body and hands are tracked in every frame on a background thread; Blender stays
   usable and a progress bar shows how far it is. The result is cached next to the video (a `FreeClipMocap` folder),
   so each video is tracked once.
4. **Check the tracking:** with **Tracked Skeleton** on, the tracker's view is drawn over the video: body green,
   left hand orange, right hand blue, points it was unsure about grey.
5. **Pick the frames** (optional): scrub the timeline to where the move starts and click the eyedropper next to
   **Start**, then do the same for **End**. Video frame N always sits at scene frame N.
6. **Solve:** press **Solve**. A new Action (a "take") goes onto the armature, under the matching video frames. The
   scene's frame rate and range are set to the video's. Earlier takes are kept.
7. **Compare:** press **Compare View**. A camera is fitted to the video and the largest 3D viewport looks through
   it, with the video behind the rig: play or scrub to see where the take strays from the actor. Set **Lens** to the
   camera you filmed with (26 mm, the default, is a phone's main camera). Turn Root Motion on before Solve if the
   actor moves around, or the rig stays put while the actor walks away.
8. **Fix by hand** (optional): press **Add IK Controls** to get a target empty at each foot and hand (and a pole
   for each knee and elbow) that follows the take. With a limb's IK slider at 1, move its empties to change the
   pose; 0 plays the solved FK keys. The sliders can be keyed.
9. **Export:** **FBX** (built like a Mixamo clip, for game engines) or **BVH**, of the current take, as it plays
   (IK included).

### The panel

| Box | Setting | What it does |
|---|---|---|
| Video | Show Video | Opens the video beside the 3D viewport (splits the largest viewport the first time) |
| | Tracked Skeleton | Draws the tracking over the video |
| | Start / End | The video frames Solve uses; End 0 means the last frame. The eyedroppers take the current frame |
| Target | Armature | The rig to animate |
| | Preset row | Apply, save or delete bone map presets; yours are kept in your FreeClipMocap user folder |
| | Guess Bone Map | Guesses the map again (replaces your edits, after asking) |
| | Bone Map | Every role and the bone that plays it; editable |
| Capture | Pose Model | Heavy (most accurate, default), Full or Lite (faster previews) |
| | Capture | Tracks every frame; with several cameras, every camera's video in turn |
| Solve | All Cameras | With several cameras: solve from all of them, triangulated through the calibration (needs Capture and Calibrate). Off: camera 1 only |
| | Smoothing | How strongly jitter is smoothed; fast moves stay sharp at any setting |
| | Root Motion | Off (default): in place, the body still moves up and down and sways. On: the body travels as in the video |
| | Fewer Keys / Tolerance | Keeps only the keys needed to stay within the tolerance (2° by default), so the take is easy to edit. Hips and legs are kept tighter so planted feet do not slide |
| | Compare View / Lens | Fits a camera (`FreeClipMocap Camera`, updated on every press) to the video and shows the take through it, video behind the rig. Sets the render size to the video's. Lens: the filming camera's focal length, 35 mm equivalent; Solve from one camera uses it too, to straighten the legs |
| IK/FK | Add IK Controls | A target (foot/hand) and a pole (knee/elbow) empty per limb, following the take, in the collection `<armature> IK Controls` |
| | Left Leg … Right Arm | Per limb: 0 = FK (the solved keys), 1 = IK (reaches for its target). Keyable |
| | Match IK to FK | Keys the targets to follow the FK keys again (replaces your target moves) |
| | Remove IK Controls | Deletes the empties and constraints; Keep the Pose bakes what you see into the FK keys first |
| Export | FBX / BVH | Exports the armature and its current take as it plays (IK included), without the mesh |

### What Solve does for you

- **Feet:** finds when a foot is planted (static camera needed), keeps it still and flat on the floor with leg IK
  and moves the hips up and down to match.
- **Hands:** fingers come from MediaPipe's hand tracker; short tracking gaps are blended, and fingers cannot bend
  in impossible ways.
- **Clean-up:** jitter is smoothed, single frames where the tracker mixed up left and right are skipped, a tilted
  camera is levelled, and body parts that are rarely visible are left unanimated (Solve tells you which).
- **Legs from the video** (one camera): the tracker's 3D often bends straight legs toward or away from the lens;
  Solve puts each leg joint back where the video shows it, using the **Lens** setting (Compare View's).
- **Rigify:** arms and legs are switched to FK and neck/head follow turned off (Solve lists each change); the keys
  go on the FK controls you animate with.

### Limits

One camera sees depth poorly: moves toward or away from the camera flatten, twists (forearm roll, turning the
torso) are weak, and the tracker may fold a straight body toward or away from the lens (bent knees, a lean). A
second camera, calibrated, fixes most of it. Fast moves blur; small hands give jittery fingers. The result is a starting point to polish by hand.

### Troubleshooting

| Problem | What to do |
|---|---|
| No FreeClipMocap tab | Press N in the 3D viewport; check the extension is enabled in Preferences → Add-ons |
| Solve says "No capture" | Press Capture first; changing the Pose Model needs a new capture |
| A required bone is red | Pick the right bone in the Bone Map list, or apply a preset |
| Legs not animated | They were out of the picture too often; film the whole body |
| Feet slide | Use a tripod; with Root Motion off, feet slide on purpose while the actor steps (an in-place walk) |
| Fingers jitter | Film at 1080p or 4K, hands bigger in the picture |
| Action Editor is empty | Select the armature: the Action Editor shows the active object's take |
| Rig in Compare View sits off the actor | Check Lens; with Root Motion off the rig stays in place while the actor travels. A rig with other proportions than the actor never lines up exactly |

## For developers

### Goals

- **Video → animation on a rig, inside Blender.** Pick a video and a target armature; the result is an Action on that
  armature, exportable as FBX or BVH.
- **Any humanoid rig:** the bone map is guessed automatically, editable, and saved as presets (Mixamo, Rigify).
- **Single camera first**, multi-camera later.
- **Local processing.** Video, landmarks and animations never leave the machine; no cloud APIs, no telemetry of
  our own. The bundled MediaPipe is unchanged from PyPI, so it sends Google usage reports (accepted 2026-11-04).
- **Easy install** as a Blender extension from extensions.blender.org (one zip per platform, dependencies and
  models bundled).
- **Latest MediaPipe** for pose and hand tracking.
- **All Python** (decided 2026-11-03), so the extension follows extensions.blender.org's rules: Python code plus
  bundled wheels, no separate programs.

### Build and test

See [Tools/README.md](Tools/README.md) for the one-time setup (wheels, models, development link) and
[Tests/README.md](Tests/README.md) for the tests. To build the zip:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe" Tools\build_extension.py
```

It writes `dist/freeclipmocap-<version>.zip`, checks it, installs it into a throwaway Blender user folder and runs
all tests against that copy.

### Documents

| File | Contents |
|---|---|
| [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md) | Parts of the extension, data flow, solver approach |
| [Docs/ROADMAP.md](Docs/ROADMAP.md) | Phases 0–4, checklists, effort estimates |
| [Docs/RESEARCH.md](Docs/RESEARCH.md) | Existing tools, MediaPipe, Blender and extensions.blender.org facts |
| [Docs/PRIVACY.md](Docs/PRIVACY.md) | Privacy rules, MediaPipe's usage reports, and how to verify them |
| [Docs/DECISIONS.md](Docs/DECISIONS.md) | Decisions made so far and open questions |
| [Docs/FILMING.md](Docs/FILMING.md) | How to film for good results (for users) |
| [Docs/FIRST_USER.md](Docs/FIRST_USER.md) | LifeInArena: the game this is built for first, its rigs and import pipeline |

### Layout

```
src/freeclipmocap/   The extension (its root is what gets zipped)
  ui/                  Panel, operators, properties (bpy)
  rig/                 Bone map, auto-guess, presets (bpy)
  capture/             OpenCV + MediaPipe → landmark cache (no bpy)
  solver/              numpy only: filters, swing/twist, grounding, finger limits (no bpy)
  animation/           Action and keyframes, FBX/BVH export (bpy)
  constants/           Shared constants, one module per topic (landmark indices, bone roles)
  presets/bone_maps/   Built-in bone map presets (JSON)
  models/, wheels/     Generated by Tools/ (wheels as on PyPI, .task models)
  THIRD_PARTY_NOTICES.md, LICENSES/   Licences of the bundled parts (ship in the zip)
Tools/               Build tools (Python, run with Blender's python.exe)
Tests/               unittest tests, mirroring the package layout
Docs/                Planning documents
dist/                Built extension zip (generated by Tools/build_extension.py)
Scratch/             Throwaway experiments and test profiles (git-ignored)
LICENSE              GPL-3.0 licence text
```

The repository folder is `FreeClipMoCap` (renamed 2026-11-03 from `BlendMotion`, the project's first name, already
taken by someone else).
