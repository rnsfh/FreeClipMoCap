# Privacy rules

The user's requirement (2026-11-02): **no uploads to third parties and no information given to third-party tools.**
One exception accepted by the user on 2026-11-04: **MediaPipe's own usage reports to Google** (see below).

## Rules

1. Video, frames, landmarks, animations and rigs never leave the machine.
2. No cloud APIs, no telemetry of our own, no update checks, no runtime downloads. The extension's own code makes
   no network connections. The manifest declares the `network` permission only because of MediaPipe's usage
   reports.
3. Models and packages are downloaded **once, at build time**, and bundled in the extension. Every URL is named to
   the user before it is fetched.
4. Before a dependency is used, its code is searched for network calls (`urllib`, `requests`, `http`, `socket`,
   analytics/usage-logging code) and the findings are reported.
5. A dependency update repeats step 4.

## Known unavoidable contacts (build time only)

- `pip download` sends a standard user-agent describing the system (OS, Python version) to PyPI. No user files.
- Model downloads from Google's storage send an ordinary HTTP request. No user files.

## Accepted third-party behaviour (while the extension runs)

- **MediaPipe's PyPI wheel sends usage reports** to `play.googleapis.com/log` from its native library
  (`libmediapipe.dll`), with no off switch (see RESEARCH.md, "MediaPipe telemetry"). Seen so far: one upload per
  pose or hand tracker, when it is closed or 60 s after it starts. Known contents: host environment (Python), OS,
  Python version; Google also sees the computer's IP address. Google's maintainers say input data (video,
  landmarks) is never sent; not verified by us.
- From 2026-11-02 to 2026-11-04 the bundled library was patched so these reports could not leave the machine.
  extensions.blender.org does not accept a modified MediaPipe, so the user chose to ship it unchanged (0.2.1).
- The reports are sent even when Blender's **Allow Online Access** is off (the user's choice, 2026-11-04). A user who
  wants none can block `blender.exe` in Windows Firewall: capture still works offline (verified 2026-11-02 with the
  patched library, which fails the same way an unreachable server does).
- matplotlib (bundled because MediaPipe imports it) writes a font cache to `%LOCALAPPDATA%\matplotlib` on first use.
  Local only.

## Verification

- **Offline test:** disconnect from the internet (or block `blender.exe` outbound in Windows Firewall), then run
  capture → solve → export. Everything must work.
- Repeat the offline test before every release.
- **Phase 0 result (2026-11-02):** with the internet disconnected, `Scratch/Phase0/scripts/offline_test.ps1`
  (generated video → MediaPipe 1.2.0 from the minimal bundle → C# probe) worked end to end. The only TCP
  connections from Blender and its child processes were 2 attempts to `127.0.0.1:1`, the patched telemetry
  endpoint (one per task, made when the task closed), refused locally. (The C# probe has since been dropped:
  all Python, 2026-11-03; the patch since 2026-11-04.)
