# Tools

Build tools, Python, run with Blender's own `python.exe`
(`C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe`). No .NET needed.

| Script | What it does |
|---|---|
| `bundled_wheels.py` | Not run directly: the list of bundled wheels with PyPI's SHA-256 for each, and the packages they install. The other tools refuse any wheel not byte-identical to PyPI's |
| `prepare_extension_files.py <wheels folder> <models folder>` | Checks first that every wheel in `bundled_wheels.py` is in `<wheels>` unchanged, the four `.task` models are in `<models>` and `blender_manifest.toml` lists exactly these wheels; then **deletes and recreates** `src/freeclipmocap/wheels/` and `models/` and copies them in |
| `build_extension.py [--blender <blender.exe>]` | **Deletes and recreates** `dist/` (refuses if it holds anything but zips), builds `freeclipmocap-<version>.zip` with Blender's own `--command extension build`, checks the zip (manifest, wheels, models, built-in presets, `THIRD_PARTY_NOTICES.md` and `LICENSES/` with the GPL text matching the repo's `LICENSE`, every wheel unchanged from PyPI, no compiled files, under the ~200 MB upload limit), installs it into a throwaway Blender user folder (`BLENDER_USER_RESOURCES`, your own Blender setup is untouched), checks that only the expected packages came from the wheels and runs all tests against the installed copy |
| `link_development_extension.py [--unlink]` | Directory junction from `%APPDATA%\Blender Foundation\Blender\5.2\extensions\user_default\freeclipmocap` to `src/freeclipmocap`. Refuses to touch anything already there that is not this link; `--unlink` removes only the link |

Development setup (once, and after a wheel or model update). `<wheels>` holds the source wheels and `<models>` the
four `.task` models; currently `Scratch\SourceWheels\` and `Scratch\SourceModels\`. Fetch the wheels with Blender's
`python.exe` (contacts pypi.org and files.pythonhosted.org; installs nothing):

```powershell
& $python -m pip download mediapipe==1.2.2 matplotlib --only-binary=:all: --platform win_amd64 --python-version 3.13 --implementation cp -d Scratch\SourceWheels
```

pip also fetches wheels we do not bundle (numpy, certifi, absl-py, …); only those in `bundled_wheels.py` are used.

```powershell
$python = "C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe"
& $python Tools\prepare_extension_files.py <wheels> <models>
& $python Tools\link_development_extension.py
```

Then enable FreeClipMocap under Edit → Preferences → Add-ons. Blender installs the wheels into
`%APPDATA%\Blender Foundation\Blender\5.2\extensions\.local\`. After a code change, restart Blender.

Planned: a wheel and model download step (until then they are fetched by hand, see above).
