# Third-party notices

FreeClipMocap is licensed under the GNU General Public License, version 3 or later (GPL-3.0-or-later), copyright
2026 rnsfh; the full text is in `LICENSES/GPL-3.0-or-later.txt`. It bundles the third-party parts below, all used on your computer only: nothing is downloaded or sent
anywhere while the extension runs.

## MediaPipe 1.2.0 (modified)

- By Google LLC, licensed under the Apache License 2.0. Bundled as `wheels/mediapipe-1.2.0-py3-none-win_amd64.whl`
  from PyPI (https://pypi.org/project/mediapipe/1.2.0/).
- **Modified by FreeClipMocap**, as section 4(b) of the Apache License asks to state. Two files are changed and
  every other file is as published on PyPI:
  - `mediapipe/tasks/c/libmediapipe.dll`: the usage-report (telemetry) address `https://play.googleapis.com/log`
    is replaced by `https://127.0.0.1:1/blocked0000`, an address of the same length that cannot be reached, so
    the library cannot send usage reports. Nothing else in the library is changed.
  - `mediapipe/tasks/python/vision/drawing_utils.py`: matplotlib is imported inside `plot_landmarks` instead of at
    the top of the module, so importing MediaPipe does not need matplotlib.
- The wheel's `RECORD` hashes are updated to match, and the same notice is in the wheel as
  `mediapipe-1.2.0.dist-info/licenses/MODIFICATIONS`. MediaPipe's own `LICENSE` and `NOTICE` (which begins with
  Google's "MediaPipe Tasks Privacy Notice") are in the same folder, unchanged.
- The changes are made by `Tools/repack_mediapipe_wheel.py` in FreeClipMocap's source repository.

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
