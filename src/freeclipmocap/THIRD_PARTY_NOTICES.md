# Third-party notices

FreeClipMocap is licensed under the GNU General Public License, version 3 or later (GPL-3.0-or-later), copyright
2026 rnsfh; the full text is in `LICENSES/GPL-3.0-or-later.txt`. It bundles the third-party parts below, all
unchanged as published. Your videos, tracking data and animations are processed on your computer only. The one
part that goes online is MediaPipe: its library sends Google usage reports while it tracks (see below).

## MediaPipe 1.2.2

- By Google LLC, licensed under the Apache License 2.0. Bundled unchanged as
  `wheels/mediapipe-1.2.2-py3-none-win_amd64.whl` from PyPI (https://pypi.org/project/mediapipe/1.2.2/).
- MediaPipe's own `LICENSE` and `NOTICE` are inside the wheel, in `mediapipe-1.2.2.dist-info/licenses/`. The
  `NOTICE` begins with Google's "MediaPipe Tasks Privacy Notice".
- **Usage reports:** MediaPipe's library (`mediapipe/tasks/c/libmediapipe.dll`) sends usage reports to Google
  (`https://play.googleapis.com/log`) when a pose or hand tracker is used. They describe the environment (such as
  the operating system and Python version). Google states that input data such as your video is not sent.
  MediaPipe offers no setting to turn this off; blocking `blender.exe` in your firewall stops it, and capture
  still works offline.

## MediaPipe models

By Google LLC, licensed under the Apache License 2.0 (as stated in each model card). Bundled unchanged in `models/`:

| File | Source | Model card |
|---|---|---|
| `pose_landmarker_lite.task` | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task | [BlazePose GHUM 3D](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf) |
| `pose_landmarker_full.task` | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task | [BlazePose GHUM 3D](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf) |
| `pose_landmarker_heavy.task` | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task | [BlazePose GHUM 3D](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf) |
| `hand_landmarker.task` | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task | [Hand Tracking (Lite/Full), Oct 2021](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf) |

The full text of the Apache License 2.0 is in `LICENSES/Apache-2.0.txt`.

## opencv-contrib-python 5.0.0.93

Bundled unchanged as `wheels/opencv_contrib_python-5.0.0.93-cp37-abi3-win_amd64.whl` from PyPI
(https://pypi.org/project/opencv-contrib-python/5.0.0.93/). Its licence texts are inside the wheel, in
`cv2/LICENSE.txt` and `cv2/LICENSE-3RD-PARTY.txt`:

- OpenCV: Apache License 2.0.
- The opencv-python packaging: MIT License, copyright Olli-Pekka Heinisuo.
- FFmpeg (`cv2/opencv_videoio_ffmpeg500_64.dll`): GNU Lesser General Public License 2.1.
- libwebp, libjpeg-turbo, quirc, Flatbuffers, Protobuf and OpenJPEG: their own permissive licences, as listed in
  `cv2/LICENSE-3RD-PARTY.txt`.
- Intel Integrated Performance Primitives (IPP ICV), statically linked: Intel Simplified Software License
  (October 2022).

## matplotlib 3.11.2 and its dependencies

MediaPipe imports matplotlib, so it is bundled with the packages it needs, each unchanged from PyPI
(`https://pypi.org/project/<name>/<version>/`). Each wheel carries its licence text in its `.dist-info` folder.

| Wheel | Licence |
|---|---|
| `matplotlib-3.11.2-cp313-cp313-win_amd64.whl` | Matplotlib License (based on the PSF License); its fonts under the DejaVu and STIX licences (`matplotlib/mpl-data/fonts/ttf/`). Statically links FreeType (FreeType License) and Qhull (Qhull License) |
| `contourpy-1.4.0-cp313-cp313-win_amd64.whl` | BSD 3-Clause |
| `cycler-0.12.1-py3-none-any.whl` | BSD 3-Clause |
| `fonttools-4.68.0-cp313-cp313-win_amd64.whl` | MIT |
| `kiwisolver-1.5.1-cp313-cp313-win_amd64.whl` | BSD 3-Clause |
| `pillow-13.0.0-cp313-cp313-win_amd64.whl` | MIT-CMU; the libraries it bundles (FreeType, HarfBuzz, libjpeg-turbo, libpng, and others) under their own licences, all included in its `LICENSE` |
| `pyparsing-3.3.3-py3-none-any.whl` | MIT |
| `python_dateutil-4.0.0-py3-none-any.whl` | BSD 3-Clause |
| `tzdata-2026.6-py2.py3-none-any.whl` | Apache License 2.0 (the time zone data itself is in the public domain) |
