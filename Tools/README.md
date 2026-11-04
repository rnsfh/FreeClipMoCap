# Tools

Build tools, Python, run with Blender's own `python.exe`
(`C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe`). No .NET needed.

| Script | What it does |
|---|---|
| `repack_mediapipe_wheel.py <source wheel> <output folder>` | Copies a mediapipe wheel with both patches: the telemetry endpoint in `libmediapipe.dll` replaced by the same-length `https://127.0.0.1:1/blocked0000`, and `drawing_utils.py` importing matplotlib inside `plot_landmarks` only. Adds a notice of both changes as `mediapipe-<version>.dist-info/licenses/MODIFICATIONS` (Apache-2.0 §4(b)). Updates `RECORD` and checks every hash in it. Accepts an already-repacked wheel (the result is identical). Stops if the files do not look as expected (a new MediaPipe version needs a look by hand). Same name and version as the source wheel |
| `prepare_extension_files.py <wheels folder> <models folder>` | **Deletes and recreates** `src/freeclipmocap/wheels/` and `models/`, then fills them: repacked mediapipe, opencv-contrib-python, the four `.task` models. Checks that `blender_manifest.toml` lists exactly these wheels |
| `build_extension.py [--blender <blender.exe>]` | **Deletes and recreates** `dist/` (refuses if it holds anything but zips), builds `freeclipmocap-<version>.zip` with Blender's own `--command extension build`, checks the zip (manifest, wheels, models, built-in presets, `THIRD_PARTY_NOTICES.md` and `LICENSES/` with the GPL text matching the repo's `LICENSE`, the mediapipe wheel patched with its notice, no compiled files, under the ~200 MB upload limit), installs it into a throwaway Blender user folder (`BLENDER_USER_RESOURCES`, your own Blender setup is untouched), checks that only `mediapipe` and `cv2` came from the wheels and runs all tests against the installed copy |
| `link_development_extension.py [--unlink]` | Directory junction from `%APPDATA%\Blender Foundation\Blender\5.2\extensions\user_default\freeclipmocap` to `src/freeclipmocap`. Refuses to touch anything already there that is not this link; `--unlink` removes only the link |

Development setup (once, and after a wheel or model update). `<wheels>` holds the source wheels (the original
mediapipe wheel and opencv-contrib-python, fetched with `pip download` by Blender's `python.exe`) and `<models>` the
four `.task` models; until 2026-11-04 these lived in `Scratch\Phase0\`, which has since been deleted. Do not point
`prepare_extension_files.py` at an empty folder: it deletes the prepared copies in `src/freeclipmocap/` first.

```powershell
$python = "C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe"
& $python Tools\prepare_extension_files.py <wheels> <models>
& $python Tools\link_development_extension.py
```

Then enable FreeClipMocap under Edit → Preferences → Add-ons. Blender installs the two wheels into
`%APPDATA%\Blender Foundation\Blender\5.2\extensions\.local\`. After a code change, restart Blender.

Planned: a wheel and model download step (until then they are fetched by hand, see above).
