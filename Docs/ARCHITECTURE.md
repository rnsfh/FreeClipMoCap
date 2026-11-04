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
  5. **Compare**: Show Video (the video beside the viewport, with the tracking drawn on it) and Compare View (a
     camera fitted to the video, video behind the rig), both synced to the timeline, to scrub and spot capture and
     solve errors.
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

0. **Legs from the video** (one camera only, `solver/leg_depth.py`, 2026-11-04; Solve does it before the solver).
   MediaPipe's one-camera 3D folds straight legs toward or away from the lens. Per frame, MediaPipe's hip-centred
   skeleton is placed in front of the lens (the shift that best puts its points on their camera rays, linear least
   squares; focal length from the panel's Lens). Each leg bone's length is the 75th percentile over the clip of its
   length laid in that depth plane (a bone never looks longer than it is). Then the hips, knees, ankles, heels and
   toes are rebuilt from the hips' centre outward, each on its camera ray at its bone's length from its parent: of
   the two places where the ray meets that sphere, the one nearer MediaPipe's guess (a ray that misses: the bone
   points at its nearest point). Torso, arms and head keep MediaPipe's 3D (repairing them made them worse). On the
   user's phone take, against the two-camera triangulation: shins within 2-5° (MediaPipe 23-25°), thighs 4-5°,
   feet 5° (MediaPipe up to 37°).
1. **Clean the points.** Body points with visibility < 0.5 count as missing in that frame; a point visible in under
   25% of the clip is not used at all (its bones are not animated, and the Solve message says so). Frames whose hip
   or shoulder line turns > 100° against the surrounding ±7 frames are tracker left/right mix-ups and are skipped
   (hands too, as they are matched through the wrists). Missing values hold the last good one (the first good one
   before it).
2. **Smooth** with the One Euro filter (slider 0–1).
3. **Into the rig's space.** MediaPipe's axes (x = image right, y = down, z = away from the camera) are mapped onto
   the rig's own right/up/back axes, found from its legs and head, so any rig orientation works. The actor faces the
   camera = the rig faces its front.
4. **Level the camera.** The median direction from the ankles to the shoulders over the clip is turned to straight
   up (a phone pointing down at the actor makes the body lean in camera space); the angle is reported. The whole
   body, not the torso (2026-11-04): a real bend at the hips stays in the torso instead of slanting the legs. A clip
   crouched or lying most of the time is levelled by that pose. Without visible ankles, the torso. Off with several
   cameras (`SolveSettings.level_by_body`), whose floor is level already.
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
     knee kept in its plane) reaches it, blended in and out over 3 frames. The lock may turn a knee at most 3°
     away from the tracked bend (near a straight knee a few mm of reach cost many degrees); the foot then stays that
     little off its target. A planted foot is turned flat (its rest
     tilt, MediaPipe's heading), so the ankle sits at its rest height; the toes turn with it. When a straight leg
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
- **Compare View** (`ui/compare_view.py`, fit in `solver/camera_fit.py`, 2026-11-03): one fixed (tripod) camera
  that shows the take where the video shows the actor. The tracking table's `joint_points` pair rig joints with
  tracked points (MediaPipe: shoulders, elbows, wrists, hips, knees, ankles, each the head of its role's bone).
  Up to 60 frames of the take are sampled; at each, the posed bone heads (evaluated by Blender, so constraints count)
  are paired with the tracked image points (visibility >= 0.5). The fit (numpy, Levenberg-Marquardt with a numeric
  Jacobian and Huber-like weights beyond 2% of the video height, 8 starting directions around the rig) finds the
  camera's position and aim; the **focal length is given** (panel field Lens, 35 mm equivalent, default 26 mm),
  because from one camera it trades off against distance: on the user's clip any lens from 20 to 70 mm aligned
  equally well (median 28-30 px on 720p), and a free fit put a 98 mm camera 9 m away below the floor. With 26 mm the
  camera lands 2.5 m in front of the rig, 0.4-0.5 m up. The camera (`FreeClipMocap Camera`, remembered in the scene
  properties and updated on each press) uses Blender's 36 mm sensor on the long side, so its Lens equals the
  setting, and shows the movie clip as a background image (stretched to the frame; the render size is set to the
  video's so the frame has the video's shape). The largest 3D viewport looks through it as a **local camera**, so
  the scene's active camera is not changed. The Info line reports the median joint-to-point distance.
- **Edit Mode:** Blender does not evaluate the pose of an armature in Edit Mode, so a take on it does not play (and
  Solve's constraint check flagged every bone). Solve and Compare View switch the target armature to Object Mode
  first and Solve says so (the user, 2026-11-03; LifeInArena's `human_male.blend` is saved in Edit Mode).
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

### IK/FK (`animation/ik_controls.py`, `ui/ik_operators.py`, 2026-11-04)

- **Add IK Controls** (needs a take): per mapped limb (legs: UpLeg/Leg/Foot, arms: Arm/ForeArm/Hand) a target empty
  `<armature> IK <limb>` and a pole empty `<armature> Pole <limb>`, parented to the armature, in the collection
  `<armature> IK Controls`. The shin/forearm gets `FreeClipMocap IK` (chain 2, target, pole, no stretch), the
  foot/hand `FreeClipMocap IK Rotation` (Copy Rotation of the target). Both influences are driven (plain Average
  driver, no Python) by the armature's custom property `IK <limb>`: 0 = FK, 1 = IK, keyable.
- **Targets follow FK:** the take is played once with the IK muted; the target is keyed at the shin/forearm tail
  with the foot/hand rotation, the pole beside the knee/elbow (key reduction within 0.1% of the limb length and
  0.25°). Blender's IK turns the chain so that the root bone's axis at the pole angle points, across the
  root-to-target line, like the pole seen from the target; the pole angle is fixed from the upper bone's bend
  axis (rig front for knees, back for elbows) and the pole is placed out from that line along the same axis, so
  the FK pose is exactly an IK solution (within 0.25° and 1 mm on the user's rig).
- **Solve** mutes the IK while writing (the constraint matching would otherwise fight it) and keys the targets to
  the new take. **Match IK to FK** keys them again (after FK edits; replaces target edits, after asking).
  **Remove IK Controls** deletes the empties, constraints, drivers and sliders; *Keep the Pose* (default) first
  bakes the evaluated pose into the FK keys of every limb IK actually changed (> 0.05°), untouched limbs keep
  their keys.
- **Export:** FBX and BVH read Blender's evaluated pose, so IK is exported as shown. FBX shifts the IK targets'
  keys with the take's while it starts the take at frame 0.
- Blender's IK starts from the pose it solved last, so a frame can differ by a few hundredths of a degree
  depending on the frame before.

Kept free of `bpy` (decided 2026-11-03), so it is tested on its own with Blender's `python.exe`, without Blender's UI.

- **Solver:** landmark points → per-bone rotations against the target rig's rest pose.
- **Filters:** jitter smoothing (e.g. One Euro filter), outlier and tracking-failure repair (gaps, swapped
  left/right limbs).
- **Feet:** contact detection (speed + height), foot locking, ground plane.
- **Root motion:** hip trajectory, or in place.
- **Multi-camera** (Phase 4, in progress): see "Multi-camera" below.

Capture runs on a background thread inside Blender (2026-11-03; to confirm in the UI on a real clip, with a
separate process of Blender's Python as the fallback). Whether the solver needs a thread too is decided once it
exists and its speed is known.

### Multi-camera (Phase 4, in progress)

Built so far (2026-11-03):

- **Panel** (`ui/calibration_operators.py`, `ui/phone_cameras.py`): the Video section lists the cameras (camera 1 is `video_path`, cameras 2+ the `extra_videos` collection with their Sync offsets); Add Camera, remove, Sync by Sound. With two or more cameras a Calibration box: Method (Board or Body; Body shows Phones Apart instead of the board settings), Save Board Image (A4 PNG), Square (mm), Paper (A4, Letter or Custom with Below Squares in mm), a Lens menu per camera (Measure, Typed with a focal length in mm and Keep Fixed, or a saved lens, with Save Lens and Delete Lens; the last calibration's focal length shown in mm), Separate Calibration Videos, Start/End, Calibrate (background job `capture/calibration_job.py` with one progress bar over the cameras, Cancel). The result is applied as one Blender camera per phone (`FreeClipMocap Phone N`) with its take video behind it. Show Video stacks one Movie Clip Editor per camera, each clip started by its Sync offset (`ui/video_operators.py`, shared with the phone cameras).

- **Sync** (`solver/audio_sync.py`, `ui/video_audio.py`): each video's sound is read with Blender's `aud` module and turned into a 1 kHz onset curve (rises in loudness, so gain and sample rate do not matter); the curves are cross-correlated and the peak gives how much later the other camera started (within ~1 ms on test sounds). A clap at the start makes the peak sharp. Loudness below 10× the take's median makes no onsets, so room hiss in a quiet take does not swamp the clap (2026-11-04).
- **Board** (`capture/board_detection.py`): ChArUco 4×5, 48 mm nominal, `DICT_4X4_50`; the page image to print (A4, 300 dpi); `detect_board` reads a frame range of a video and records, per frame, the 12 inner corners (NaN where unseen) with the video's own timestamps (`capture/board_sightings.py`).
- **Saved lenses** (`capture/saved_lens.py`, `ui/lens_folder.py`): `SavedLens` (name, video size, camera matrix, distortion, fit error), one JSON per lens in the extension's user folder under `lenses/`, saved from the last calibration's camera.
- **Calibration** (`capture/camera_calibration.py`): each lens from its own board views (or a saved lens, kept fixed; or a typed focal length, kept fixed or used as the starting point), then each further camera against the first: its corners interpolated to the first camera's frame times (start delay from the sound plus a board-based correction of up to ±60 ms), both lenses and the relative pose refined together by OpenCV's `stereoCalibrate`. The floor from a still board standing on it (its bottom edge the given distance below the squares) or lying on it; world axes: Z up, Y the first camera's level view direction, origin on the floor below where the view axes meet.
- **Calibration from the body** (`capture/body_calibration.py`): the background job loads each video's landmark cache (or captures it first); 16 body joints of each camera interpolated to camera 1's frame times, essential matrix (RANSAC) for each further camera against camera 1, the sound offset refined within ±60 ms, left/right swaps undone per frame, cameras 3+ by PnP on the first pair's triangulated joints, then a joint refinement (linear triangulation inside, Cauchy loss, camera 2 held at the measured distance). Lenses as typed or saved, never refined. Floor from the planted feet.
- **Result** (`capture/camera_setup.py`): `CameraSetup`, one `CalibratedCamera` per video (OpenCV camera matrix and 5 distortion values, world→camera rotation and translation in OpenCV axes: x right, y down, z forward), how the floor was found and its frames, the fit errors. Saved as JSON next to the first camera's video.

- **Capture** (`ui/capture_operators.py`): every camera's video in turn, one cache per video (a camera already captured with the same settings is skipped).
- **Triangulation** (`solver/triangulation.py`, numpy, 2026-11-04): camera 1's frames are the time base; each camera's 2D points are read at the same moment (Sync offset plus the calibration's picture delay, interpolated between its frames, NaN across dropped frames), lens distortion undone (the inverse of OpenCV's model, by iteration), left/right swaps of each further camera against camera 1 undone per frame (the labelling whose triangulation fits better by half), then linear triangulation of every point; points a camera sees more than 3% of its picture height away (or behind it) are dropped. MediaPipe's visibility is not used: the cameras' agreement decides. Hands are triangulated relative to the wrist; where only one camera has a hand, its MediaPipe hand (camera axes) is turned into the world. The result is a `LandmarkCache` in a level camera-1 frame (MediaPipe's axes: x right, y down, z away), with camera 1's pictures for the foot contacts, and a `TriangulationReport` (body share, median disagreement in px, swapped frames, hand sources).
- **Solve** with All Cameras (on by default when there are several videos): every camera's cache and a calibration matching the current videos, Sync and lenses are needed, else Solve says which; the solver runs as for one camera with leveling off (`SolveSettings.level_by_body`) and without the leg repair.

- **Floor from the take** (`capture/body_calibration.py` `with_floor_from_the_take`, 2026-11-04): a board calibration without a floor shot gets its floor from the planted feet of the take, triangulated through it, when every take video is already captured: inside the Calibrate job, or, for a saved calibration, when Calibrate is pressed again (no board detection then; the saved file is updated). The cameras are re-framed on that floor (the same world axes as a board floor) and the floor counts as 'from the feet'. Without captures the calibration says so.

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
