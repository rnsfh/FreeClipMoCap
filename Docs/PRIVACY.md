# Privacy rules

The user's requirement (2026-11-02): **no uploads to third parties and no information given to third-party tools.**

## Rules

1. Video, frames, landmarks, animations and rigs never leave the machine.
2. No cloud APIs, no telemetry, no update checks, no runtime downloads. The extension declares no network permission.
3. Models and packages are downloaded **once, at build time**, and bundled in the extension. Every URL is named to
   the user before it is fetched.
4. Before a dependency is used, its code is searched for network calls (`urllib`, `requests`, `http`, `socket`,
   analytics/usage-logging code) and the findings are reported.
5. A dependency update repeats step 4.

## Known unavoidable contacts (build time only)

- `pip download` sends a standard user-agent describing the system (OS, Python version) to PyPI. No user files.
- Model downloads from Google's storage send an ordinary HTTP request. No user files.

## Known third-party behaviour to neutralise

- **MediaPipe PyPI wheels send usage telemetry** to `play.googleapis.com/log` from their native library, with no
  off switch (see RESEARCH.md, "MediaPipe telemetry"). Must be neutralised before MediaPipe is ever run.

## Verification

- **Offline test:** disconnect from the internet (or block `blender.exe` outbound in Windows Firewall), then run
  capture → solve → export. Everything must work.
- Repeat the offline test before every release.
- **Phase 0 result (2026-11-02):** with the internet disconnected, `Scratch/Phase0/scripts/offline_test.ps1`
  (generated video → MediaPipe 1.2.0 from the minimal bundle → C# probe) worked end to end. The only TCP
  connections from Blender and its child processes were 2 attempts to `127.0.0.1:1`, the patched telemetry
  endpoint (one per task, made when the task closed), refused locally. (The C# probe has since been dropped:
  all Python, 2026-11-03.)
