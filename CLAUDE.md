# FreeClipMocap

Blender extension for markerless motion capture from video, **all Python** (Blender UI, MediaPipe capture, numpy
solver, keyframing, export). The project was first called BlendMotion (name taken); the repo folder is `FreeClipMoCap`.
Read `README.md` and `Docs/` before starting work; `Docs/DECISIONS.md` lists what is settled and what is still
open.

## Working rules

1. **Always ask before decisions.** Pipeline steps, file formats, libraries, licences, what a script overwrites
   or regenerates, design choices: present options with a recommendation and wait. Small code-style choices
   covered below need no question. When a script overwrites something the user may think of as hand-made, say so
   up front.
2. **No third parties.** Video, landmarks, animations and any user data never leave the machine. No cloud APIs,
   no telemetry of our own, no runtime downloads. One-time downloads of packages and models at build time are
   allowed, each URL named to the user first. **One accepted exception** (the user, 2026-11-04): MediaPipe is
   bundled unchanged, so its native library sends Google usage reports (extensions.blender.org declined a patched
   MediaPipe). See `Docs/PRIVACY.md`.
3. **All Python** (the user's decision, 2026-11-03), so the extension is Python + bundled wheels as
   extensions.blender.org expects. No bundled programs. The build tools in `Tools/` are Python too (2026-11-03), so no C# is
   in use; the C# guidelines below apply if C# comes back.
4. **Latest MediaPipe**, the user's explicit choice (2026-11-02), even when very recent.
5. The user knows C# best and is learning Python and Blender as they go: explain what was done and why, and
   relate Python ideas to their C# counterparts where that helps.

## Environment

- Windows 11. Blender 5.2.3 LTS: `C:/Program Files/Blender Foundation/Blender 5.2/blender.exe`
  (Python 3.13.13, numpy 2.3.4 bundled). Headless: `blender.exe -b --factory-startup --python script.py -- <args>`;
  pass `--python-exit-code 1` or Blender exits 0 on a script error.
- .NET SDKs 10.0.103 / 10.0.300 / 10.0.401 (for the C# build tools).
- Standalone Python 3.15.2 installed (`python` / `py`), pip 26.2.2. It is not Blender's version: anything that must
  match Blender uses Blender's own `python.exe` (under the Blender install's `5.2/python/bin/`, pip 26.0.1), which
  can run `pip download` to fetch wheels without installing anything.
- Visual Studio Community 2026 is installed, but without the C++ tools (checked 2026-11-02; no longer needed).

## C# guidelines (build tools)

1. **`SingleOrDefault` / `Single` when exactly one match is expected.** `FirstOrDefault` / `First` only when
   several matches are normal and the first one is wanted.
2. **Records for data objects.** A type that only carries data and is never changed after it is built is a
   `sealed record` (or `readonly record struct` for small values). Types that hold changing state stay classes.
3. **`Create` factories, hidden constructors.** Every class or record of ours that code instantiates has a
   `private` constructor and a `public static T Create(...)`; callers write `Skeleton skeleton = Skeleton.Create(...)`,
   never `new Skeleton()`. Changes of state go through named methods (`Update...`, `Set...`). .NET types
   (`List<T>`) are created with `new` as usual.
4. **No comments.** Names say what code does. Comments only where code is genuinely hard to follow without one
   (non-obvious math, a workaround). No abbreviated names (`deltaSeconds`, not `dt`) unless the abbreviation is the
   common name of the thing (`Fps`, `Uv`, `Json`).
5. **Shared constants in one place:** static classes in a `Constants/` folder. A value repeated inside one class is
   a `private const` there; a value used once stays inline.
6. **Loose coupling.** Depend on interfaces and on what is passed in, not on concrete types from other areas.
7. **No `var`.** Always write the full type name (enforce `IDE0008` as a warning).

## Python

All extension code. Full, unabbreviated names and no comments, as in C#. Otherwise the same spirit as the C# rules
(user, 2026-11-03), but how a C# rule translates to Python is the user's call: whenever a Python implementation
choice comes up, ask before settling it. Settled so far (2026-11-03, details in `Docs/DECISIONS.md`):

- Data objects: `@dataclass(frozen=True, slots=True)`.
- Construct simple objects directly; a named `@classmethod` when building does real work
  (`BoneMap.guess_from_armature(...)`). State changes through named methods.
- Shared constants in `freeclipmocap/constants/`, one module per topic, `UPPER_SNAKE_CASE`; module-only ones as
  `_LEADING_UNDERSCORE` at the top of the module.
- Type hints on every parameter, return value and field (Pylance basic).
- `solver/` imports only numpy, `capture/` only mediapipe + cv2: no `bpy` in either.
- Tests: `unittest`, run with Blender's `python.exe` (or headless `blender.exe` for bpy parts).

## Blender extension rules (extensions.blender.org add-on guidelines)

- No inserting modules into `sys.modules` and no changing `sys.path` / module loading; third-party code only as
  bundled wheels.
- Never write into the add-on's own folder; use `bpy.utils.extension_path_user(__package__, create=True)`.
- No installing packages at runtime, no auto-updater, no dependence on external servers.
- Licence: GPL-3.0-or-later for add-ons (and chosen for the whole project, 2026-11-03).
