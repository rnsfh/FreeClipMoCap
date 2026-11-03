from pathlib import Path

import cv2
import numpy as np


def write_moving_shapes_video(path: Path, frame_count: int, frame_rate: float = 30.0) -> Path:
    width = 640
    height = 480
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height))
    for frame_index in range(frame_count):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        x = 80 + (frame_index * 7) % (width - 160)
        cv2.circle(frame, (x, 240), 40, (60, 180, 240), -1)
        cv2.rectangle(frame, (width - x - 30, 100), (width - x + 30, 160), (200, 80, 80), -1)
        writer.write(frame)
    writer.release()
    return path
