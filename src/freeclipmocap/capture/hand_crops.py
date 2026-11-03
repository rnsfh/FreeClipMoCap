from dataclasses import dataclass

import cv2
import numpy as np

SMALLEST_CROP_PIXELS = 256
_SIDE_PER_HAND_LENGTH = 15.0
_SIDE_PER_TORSO_LENGTH = 2.2
_SMALLEST_SIDE_PIXELS = 96.0


@dataclass(frozen=True, slots=True)
class CropBox:
    left: float
    top: float
    side: float

    @classmethod
    def around_hand(cls, wrist: np.ndarray, knuckles: np.ndarray, torso_pixels: float) -> "CropBox":
        hand = knuckles - wrist
        center = wrist + 0.5 * hand
        side = max(
            _SIDE_PER_HAND_LENGTH * float(np.linalg.norm(hand)),
            _SIDE_PER_TORSO_LENGTH * torso_pixels,
            _SMALLEST_SIDE_PIXELS,
        )
        return cls(left=float(center[0]) - side / 2.0, top=float(center[1]) - side / 2.0, side=side)

    @property
    def pixels(self) -> int:
        return max(SMALLEST_CROP_PIXELS, int(np.ceil(self.side)))

    def cut(self, frame: np.ndarray) -> np.ndarray:
        scale = self.pixels / self.side
        to_crop = np.array([[scale, 0.0, -self.left * scale], [0.0, scale, -self.top * scale]])
        return cv2.warpAffine(
            frame, to_crop, (self.pixels, self.pixels), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

    def to_crop(self, frame_pixels: np.ndarray) -> np.ndarray:
        return (frame_pixels - (self.left, self.top)) / self.side

    def to_frame(self, crop_landmarks: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
        frame_landmarks = np.empty_like(crop_landmarks)
        frame_landmarks[:, 0] = (self.left + crop_landmarks[:, 0] * self.side) / frame_width
        frame_landmarks[:, 1] = (self.top + crop_landmarks[:, 1] * self.side) / frame_height
        frame_landmarks[:, 2] = crop_landmarks[:, 2] * self.side / frame_width
        return frame_landmarks
