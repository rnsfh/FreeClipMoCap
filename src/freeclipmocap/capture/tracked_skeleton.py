from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..constants import hand_landmarks, pose_landmarks
from .landmark_cache import LandmarkCache


class SkeletonPart(Enum):
    BODY = "body"
    HIDDEN_BODY = "hidden_body"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"


@dataclass(frozen=True, slots=True, eq=False)
class SkeletonLines:
    part: SkeletonPart
    segments: np.ndarray


def tracked_skeleton_lines(cache: LandmarkCache, frame_index: int, minimum_visibility: float) -> list[SkeletonLines]:
    if not 0 <= frame_index < cache.frame_count:
        return []
    lines: list[SkeletonLines] = []
    if cache.pose.found[frame_index]:
        points = cache.pose.image_landmarks[frame_index, :, :2]
        visible = cache.pose.visibility[frame_index] >= minimum_visibility
        connections = np.array(pose_landmarks.CONNECTIONS)
        both_visible = visible[connections[:, 0]] & visible[connections[:, 1]]
        lines.append(SkeletonLines(SkeletonPart.BODY, points[connections[both_visible]]))
        lines.append(SkeletonLines(SkeletonPart.HIDDEN_BODY, points[connections[~both_visible]]))
    for part, track in ((SkeletonPart.LEFT_HAND, cache.left_hand), (SkeletonPart.RIGHT_HAND, cache.right_hand)):
        if track.found[frame_index]:
            points = track.image_landmarks[frame_index, :, :2]
            lines.append(SkeletonLines(part, points[np.array(hand_landmarks.CONNECTIONS)]))
    return [skeleton_lines for skeleton_lines in lines if len(skeleton_lines.segments)]
