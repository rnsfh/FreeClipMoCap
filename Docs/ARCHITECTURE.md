# Architecture

Status: agreed in outline; switched to all Python on 2026-11-03. Details marked *open* are listed in
[DECISIONS.md](DECISIONS.md).

## Why all Python

Blender's API (`bpy`) exists only in Python, and MediaPipe is used through its Python package. The first plan put
the solver in a C# program run as a subprocess, but extensions.blender.org's rules expect an extension to be
Python code plus bundled wheels, and say nothing that allows a bundled native program. To make the one-click
install from extensions.blender.org certain, everything is Python (decided 2026-11-03). numpy (bundled with
Blender) carries the heavy math.

## Parts

### Blender UI (`bpy`)

- Sidebar panel in the 3D viewport (N panel):
  1. **Video**: file, start/end frame.
  2. **Target**: armature to animate, with its **bone map** (see "Target rigs").
  3. **Capture**: runs MediaPipe over every frame with a progress bar; caches landmarks next to the video so
     re-solving does not rerun the model. One cache per model + settings, so a fast preview never replaces a
     heavy capture.
  4. **Solve**: runs the solver and writes an Action on the armature. Options: smoothing, foot lock, in place,
     fingers on/off. Each Solve is a new take (new Action with a fake user); earlier takes are kept.
  5. **Compare**: the video behind the rig (Blender movie clip), synced to the timeline, to scrub and spot
     capture errors.
  6. **Export**: FBX (LifeInArena's settings, Blender's built-in exporter) and BVH (our own Y-up writer) of the Action.
- Ships as a Blender extension (`blender_manifest.toml`) with its wheels bundled; Blender installs them. Only
  `mediapipe` and `opencv-contrib-python` are bundled; numpy and certifi come with Blender.
- Files the extension writes (settings, bone-map presets) go to `bpy.utils.extension_path_user`, never into the
  extension's own folder.

### Capture (MediaPipe)

- Reads video frames with OpenCV and runs the Pose Landmarker and Hand Landmarker.
- Writes per-frame landmarks: 33 pose landmarks (image coordinates + metric "world" coordinates centred on the
  hips, plus visibility/presence) and 21 landmarks per hand, to a **binary** cache file: a numpy `.npz` of named
  arrays, loaded with `allow_pickle=False` (see "Landmark cache" below).
- Model files (`.task`) are bundled in the extension; nothing is downloaded at runtime.
- Runs on a background thread (`capture/capture_job.py`); a modal operator polls its progress 5 times a second and
  redraws the panel. Cancel stops it between frames and writes no cache.

#### Landmark cache

`<video folder>/FreeClipMocap/<video file name>.<pose model>-<settings hash>.landmarks.npz`, e.g.
`clip.mp4.heavy-1a2b3c4d.landmarks.npz`. Reused when the format version, the settings and the video's size and
modification time all match; otherwise captured again (the file is overwritten). F = frames read.

| Array | Shape, type | Contents |
|---|---|---|
| `format_version` | scalar int32 | 1 |
| `settings_json` | string | capture settings (pose model; hands are always tracked) |
| `video_file_size`, `video_modified_time_nanoseconds` | scalar int64 | which video file was captured |
| `video_frame_rate`, `video_width`, `video_height` | scalars | as read (after the phone's rotation flag) |
| `frame_indices` | F int32 | 0, 1, 2, … |
| `timestamps_milliseconds` | F int64 | from the video, starting at 0, strictly increasing |
| `pose_found` | F bool | body found in this frame |
| `pose_image_landmarks` | F×33×3 float32 | x, y in 0–1 of the image width/height; z relative depth |
| `pose_world_landmarks` | F×33×3 float32 | metres, centred between the hips |
| `pose_visibility`, `pose_presence` | F×33 float32 | 0–1 |
| `left_hand_*`, `right_hand_*` | `found` F bool; `image_landmarks`, `world_landmarks` F×21×3 float32; `score` F float32 | per hand |

- A missing body or hand has `found = False` **and NaN** in all its values, so it can never be read as zeros.
- Left and right are the **actor's** sides, as in MediaPipe's pose landmarks. Hands are matched to the pose wrists
  (shortest total distance in pixels), not by MediaPipe's handedness label, which assumes a mirrored selfie image.
  Hands are kept only in frames where the body was found.
- The bundled mediapipe wheel carries two patches: the telemetry endpoint in `libmediapipe.dll`, and
  `drawing_utils.py` importing matplotlib only inside `plot_landmarks` (so matplotlib is not needed).

### Solver (numpy, no `bpy`)

Table-driven (decided 2026-11-03): a table per tracking model says, for each role, which two points give the bone's
direction and which give its twist. Only MediaPipe's table exists (`solver/mediapipe_table.py`); another model would
add its own table, a capture module and new optional roles, and the cache records which model and point set it
holds.

Steps of a Solve (`solver/pose_solver.py`, built 2026-11-03):

1. **Clean the points.** Body points with visibility < 0.5 count as missing in that frame; a point visible in under
   25% of the clip is not used at all (its bones are not animated, and the Solve message says so). Frames whose hip
   or shoulder line turns > 100° against the surrounding ±7 frames are tracker left/right mix-ups and are skipped
   (hands too, as they are matched through the wrists). Missing values hold the last good one (the first good one
   before it).
2. **Smooth** with the One Euro filter (slider 0–1).
3. **Into the rig's space.** MediaPipe's axes (x = image right, y = down, z = away from the camera) are mapped onto
   the rig's own right/up/back axes, found from its legs and head, so any rig orientation works. The actor faces the
   camera = the rig faces its front.
4. **Level the camera.** The median torso direction over the clip is turned to straight up (a phone pointing down
   at the actor makes the body lean in camera space); the angle is reported.
5. **Solve each mapped bone, parents first**, against the rig's rest pose: *swing* (turn the bone onto the measured
   direction, keeping the twist it inherits), *frame* (direction + a second line fixes the twist: hips by the hip
   line, hands by index→pinky knuckles, head by ear→eye with a 10° anatomical offset and the ear line), or
   *bending limb* (upper arm and thigh take their twist from the elbow/knee bend, blended out as the limb
   straightens). The spine bones share the turn between the hips and the shoulder line evenly. Fingers are solved
   in the hand's own frame from the Hand Landmarker, so a hand-model / body-model mismatch does not bend them.
6. **Resample** to the scene's frame rate, which Solve first sets to the video's (nearest standard rate); write a new
   Action in each bone's own rotation mode.
   Before resampling, **grounding** (`solver/grounding.py`, numpy; the table's `foot_rules` name the leg roles and the
   landmarks of each foot's contact point):
   - *Planted feet*: heel + toe nearly still in the image (under 0.6 torso lengths per second; assumes a static
     camera) and not lifted above the other foot (under 0.12 torso lengths); gaps of up to 2 frames filled, contacts
     shorter than 3 frames dropped.
   - *Floor*: the rig's own rest-pose foot height; "up" is the armature axis nearest to the body's up. A foot's
     distance to the floor is its lowest point (ankle, ball, toe tip) minus that point's rest height.
   - *Hips height*: every planted foot reaches the floor (the higher one sets it; the IK lifts the other); between
     contacts it is interpolated; then smoothed with the clip's smoothing. *Hips travel*: a planted foot does not
     move, so the body moves by the opposite of its step.
   - *Foot lock*: each planted stretch holds the ankle at its median spot on the floor; two-bone IK (thigh + shin,
     knee kept in its plane, foot orientation kept) reaches it, blended in and out over 3 frames. When a straight leg
     cannot reach, the hips are lowered just enough.
   - *In place* (Root Motion off): each foot's planted spot is followed over the clip (moving only when that foot
     lands somewhere new); the slow drift of their average (1.5 s window) is taken off the hips, so leans, crouches
     and sways stay and only stepping travel goes.
   - Output: the hips' location keys (`SolvedTake.locations_by_bone`) and the IK'd leg rotations. Legs too rarely
     visible: no grounding, the hips stay at rest height (reported).
7. **Fewer keys** (on by default, 2°; hips location within 0.2% of the hips height; the hips and each bone above a
   foot get a tolerance that moves the foot by at most 0.2% of the hips-to-toe length, so planted feet stay within
   about 5 mm instead of sliding several cm): keys are picked by
   Douglas–Peucker on each bone's rotation (and position), then the curves
   are evaluated by Blender and keys added wherever the result strays beyond the tolerance, until every frame is
   within it (`animation/key_reduction.py`, `animation/action_writer.py`).

Solve works on the video frames chosen in the panel (Start/End, End 0 = last frame; `LandmarkCache.sliced`), so
capture always covers the whole video once and changing the range needs no new capture. Times are counted from the
first chosen frame. Video frame N always sits at scene frame N, so the take is keyed under the same video frames
and Solve sets the scene range to them; the Set buttons (eyedropper) copy the current frame into Start or End.

### Show Video and export (`ui/video_operators.py`, `animation/export.py`)

- **Show Video** loads the video as a movie clip starting at scene frame 1 and shows it in a Movie Clip
  Editor beside the largest 3D viewport (splitting it the first time); playback is set to drop frames so both run
  in real time, and clip editors are redrawn during playback (off in a fresh Blender, which froze the video). The
  clip editor's toolbar and sidebar are hidden (in a fresh editor they take ~490 px and squeezed the video to 2 px),
  and the video is fitted to the area a moment later, once Blender has laid it out. (Blender's `clip.prefetch`
  cannot be called from a script, so it is not used.)
- The clip editor's view coordinates are **normalized** (0–1 across the video frame); the overlay maps MediaPipe's
  image coordinates (0–1, y down) with `view_to_region(x, 1 − y)`. Checked in a real window on 2026-11-03: the
  frame's corners land on the video's corners.
- Solve selects the target armature and makes it active, so the Action Editor (which shows the active object's
  action) shows the new take at once.
- **Tracked Skeleton** (checkbox, on by default): a draw handler on the Movie Clip Editor (`ui/video_overlay.py`)
  draws the raw tracking of the current frame over the video, straight from the cache: body green, left hand
  orange, right hand blue, lines to points the tracker marked hidden grey, nothing where tracking was lost. It
  shows what MediaPipe saw, not the solved animation, so capture errors are visible at a glance.
- **FBX:** the target armature and its current take only, baked over the scene range, no leaf bones, built like a
  Mixamo clip (centimetre units, metre-sized values; see DECISIONS.md "FBX scale settings"). The FBX take is named
  after the take and starts at 0 s: during the export the take's keys and the scene range are shifted to start at
  frame 0 and the scene is renamed to the take (Blender names the take after the scene), then all three are restored.
- **BVH:** our own writer (`animation/bvh_writer.py`, numpy only) over the scene range: Y-up, metres, position
  channels on the root only, Z-X-Y rotation channels, joints in the armature's hierarchy, `End Site` from leaf bone
  tails. `export.py` samples each bone's pose matrix per frame; the writer turns them into rotations relative to
  the parent (BVH joints have no rest orientation) and converts Blender's Z-up to Y-up.
- Both open Blender's file dialog, proposing `<take name>.fbx/.bvh` next to the video; an existing file shows red
  and the user confirms overwriting.

Kept free of `bpy` (decided 2026-11-03), so it is tested on its own with Blender's `python.exe`, without Blender's UI.

- **Solver:** landmark points → per-bone rotations against the target rig's rest pose.
- **Filters:** jitter smoothing (e.g. One Euro filter), outlier and tracking-failure repair (gaps, swapped
  left/right limbs).
- **Feet:** contact detection (speed + height), foot locking, ground plane.
- **Root motion:** hip trajectory, or in place.
- **Later, multi-camera:** calibration (OpenCV), video sync, triangulation (see Phase 4 in [ROADMAP.md](ROADMAP.md)).

Capture runs on a background thread inside Blender (2026-11-03; to confirm in the UI on a real clip, with a
separate process of Blender's Python as the fallback). Whether the solver needs a thread too is decided once it
exists and its speed is known.

### Target rigs

The solver works with **roles** (hips, spine, left upper arm, right index finger joint 2, …). A **bone map** says
which bone of the armature plays each role.

Roles carry Mixamo names (`constants/bone_roles.py`, 52 roles: 22 body, 15 fingers per hand). They are our own
names, not MediaPipe's: MediaPipe returns numbered points only, and the solver turns each pair of neighbouring
points into the direction of one role's bone (e.g. hand points 1→2, 2→3, 3→4 drive Thumb1, Thumb2, Thumb3; the
fourth Mixamo finger bone is the tip and is never keyed). The map lives on the
armature object (`object.freeclipmocap_rig.bone_map`, one entry per role), so it is saved in the .blend.

1. **Auto-guess** (`rig/bone_guess.py`, no `bpy`): when an armature is picked whose map is still empty, every preset
   is matched (a namespace prefix such as `mixamorig:` is ignored); a preset counts only when it covers every
   required role, and the one covering the most roles is taken; roles it leaves empty
   are guessed from bone names (side from `Left`/`L`/`.l`/`_r` …, kind from `upperarm`/`thigh`/`calf`/`clavicle`
   …; tip, twist, IK and metacarpal bones skipped) and depth in the hierarchy (spine bones in order, finger joints
   1–3). Picking the armature again never overwrites the map; **Guess Bone Map** and **Apply Preset** do, after a
   confirmation.
2. **Editable map:** the panel lists each role with a search field of the armature's bones; missing required roles
   are shown in red.
3. **Presets:** JSON `{format_version, name, bones_by_role, rig_settings?}`. Built-in ones in
   `freeclipmocap/presets/bone_maps/` (Mixamo, Rigify); the user's saved with **Save Preset** into the extension's
   user folder (`extension_path_user(...)/bone_map_presets/`) and removed with **Delete Preset** (after a
   confirmation; built-in presets cannot be deleted).
4. **Control rigs** (Rigify): the map names the control bones the animator works with (`torso`, `spine_fk.001`…,
   `upper_arm_fk.L`, `thigh_fk.L`, `f_index.01.L`, …), not the deform bones. A preset's optional `rig_settings`
   (`{bone: {custom property: value}}`) are the rig's own sliders that Solve sets first; for Rigify: arms and legs to
   FK (`IK_FK` = 1 on `upper_arm_parent.*` / `thigh_parent.*`) and neck/head follow off. Guessing or applying a
   preset copies them onto the armature (`freeclipmocap_rig.rig_settings_json`); Solve reports each change.
5. **Rigs whose bones are moved by constraints:** the solver assumes unmapped bones in between stay at rest, which
   is not true in a control rig (Rigify's arm helpers follow the shoulder control, for example). The action writer
   therefore lets Blender evaluate the rig with all its constraints after writing every frame, and corrects each
   animated bone whose orientation differs from the solver's by more than 0.05°, pass by pass (each pass fixes one
   more level of the chain); then key reduction runs on the corrected values. Plain rigs pass the first check. On the
   user's clip a generated Rigify rig needed 3 passes (1.5 s) and then matched every control within 0.01°. Bones
   still off after 12 passes are reported.

## Data flow

```
video ──► capture (OpenCV + MediaPipe) ──► landmark cache, binary (next to the video)
                                                 │
target armature ──► rest pose + bone map ──►  solver: filter → solve → feet → root
                                                 │
                                                 ▼
                                     keyframes → Action on the armature ──► FBX / BVH export
```

## Solver approach (first version)

- MediaPipe world landmarks are metric, centred between the hips, so body shape comes from them; image coordinates
  help with root travel.
- Each bone's rotation = **swing** (rotate the rest-pose bone direction onto the measured direction) + **twist**
  from a second reference direction:
  - hips: left hip → right hip
  - chest: left shoulder → right shoulder
  - head: ears / nose / eyes
  - hands: index and pinky knuckles (and the hand model when present)
- Rotations are computed against the target rig's own rest pose, so any mapped rig works without retargeting.
- Fingers from the Hand Landmarker, run on a crop around each wrist; gaps are blended between the good poses either
  side, and finger joints are clamped to human limits (`solver/finger_limits.py`).

## Known single-camera limits

- Depth is rough: moves toward or away from the camera flatten.
- Twists are weak: forearm roll, torso turning toward the camera.
- Occlusion: the body hiding an arm, turning the back to the camera, floor work.
- Fast motion blurs. Filming at 60 fps helps.
- Proportions: motion copies rotations, so contacts (hands meeting, hand on hip) can miss on a body shaped unlike
  the actor.
