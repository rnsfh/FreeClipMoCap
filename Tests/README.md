# Tests

`unittest`, mirroring the package layout. Everything runs in headless Blender through `run_in_blender.py`, which
enables the linked extension (run `Tools/link_development_extension.py` first), checks that enabling it did not load
mediapipe or cv2, then discovers every `test_*.py` here:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b --factory-startup --python-exit-code 1 `
  --python Tests\run_in_blender.py
```

Tests import the extension as `bl_ext.user_default.freeclipmocap...`, the name Blender gives it. Videos are generated
(moving shapes, no person) into a temporary folder; no footage of yours is used.

| Test | Checks |
|---|---|
| `test_development_install.py` | Extension enables; panel registered; mediapipe and cv2 load from the extension's wheels; telemetry endpoint blocked; vision loads without matplotlib; models bundled |
| `capture/test_landmark_cache.py` | Cache path and naming; save/load keeps every value (NaN too); loaded arrays read-only; reuse only while video and settings match; other format versions refused |
| `rig/test_bone_guess.py` | Guess on Mixamo (with, without and with another namespace), Unreal-style and metarig-style names; tips, `headfront` and tails stay unmapped; preset first, names fill the rest |
| `rig/test_bone_map_preset.py` | Preset save/load; a broken preset is reported while others still load; unsafe names refused; namespace-free matching only when unique; merging never reuses a bone |
| `rig/test_armature_bone_map.py` | On a real armature: picking it guesses the map, picking again keeps hand edits, Guess and Apply Preset operators, entries naming missing bones ignored |
| `capture/test_landmark_capture.py` | Hands matched to the nearer wrists; every frame recorded, missing pose as NaN; background job writes the cache while the main thread keeps running (longest pause < 100 ms); cancel writes no cache; unreadable video reports an error |
| `solver/test_pose_solver.py` | On a synthetic T-pose rig with random bone rolls and points made from it: rest pose gives no rotation; a 90° elbow bend moves only the forearm; a 90° body turn only the hips; a shoulder twist spreads evenly over the spine; a tilted camera is leveled; legs out of view or seen in few frames are not animated; a briefly hidden point holds; a single left/right mix-up is skipped; a lost hand leaves its fingers unanimated; errors for no body or another tracking model; resampling keeps the duration |
| `solver/test_solver_parts.py` | Quaternion/matrix round trip, rotation between vectors (also opposite), slerp, sign continuity, hold-last-good, One Euro filter |
| `animation/test_action_writer.py` | Quaternion and Euler bones keyed in their own mode; reduced keys stay within the tolerance on both; each write is a new take (`Take`, `Take.001`) with a fake user; the Solve button end to end on a synthetic capture sets the scene to the video frame rate, reduces still bones to 2 keys and bends the elbow by 90° |
| `animation/test_export_and_frames.py` | Phone frame rates snap to standard ones; key selection keeps the ends and turning points; FBX built like a Mixamo clip (centimetre units, no ×100 node, Hips at metre values); BVH has every frame; the video clip for Show Video starts at the take. (Splitting the viewport needs a window and is checked by hand) |
| `capture/test_tracked_skeleton.py` | Overlay lines for Show Video: visible body and found hands drawn, lines to hidden points kept apart, nothing for frames without a body or outside the clip. (The drawing itself needs a window and is checked by hand) |
