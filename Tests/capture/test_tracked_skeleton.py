import unittest

import numpy as np

from bl_ext.user_default.freeclipmocap.capture.tracked_skeleton import SkeletonPart, tracked_skeleton_lines
from bl_ext.user_default.freeclipmocap.constants import hand_landmarks, pose_landmarks

from ..solver.synthetic_actor import make_cache, make_rig, rest_landmarks


class TrackedSkeletonTests(unittest.TestCase):
    def setUp(self) -> None:
        rig, _bone_map = make_rig()
        rest = rest_landmarks(rig)
        self.cache = make_cache([rest, rest, rest], left_hand_found=False)
        self.cache.pose.image_landmarks[:] = np.random.default_rng(1).random(self.cache.pose.image_landmarks.shape)
        self.cache.right_hand.image_landmarks[:] = 0.5

    def parts(self, frame_index: int) -> dict[SkeletonPart, np.ndarray]:
        return {lines.part: lines.segments for lines in tracked_skeleton_lines(self.cache, frame_index, 0.5)}

    def test_visible_body_and_found_hands_are_drawn(self) -> None:
        parts = self.parts(0)
        self.assertEqual(parts[SkeletonPart.BODY].shape, (len(pose_landmarks.CONNECTIONS), 2, 2))
        self.assertEqual(parts[SkeletonPart.RIGHT_HAND].shape, (len(hand_landmarks.CONNECTIONS), 2, 2))
        self.assertNotIn(SkeletonPart.LEFT_HAND, parts)
        self.assertNotIn(SkeletonPart.HIDDEN_BODY, parts)
        np.testing.assert_array_equal(parts[SkeletonPart.BODY][0], self.cache.pose.image_landmarks[0, [0, 1], :2])

    def test_lines_to_hidden_points_are_drawn_apart(self) -> None:
        self.cache.pose.visibility[1, pose_landmarks.LEFT_KNEE] = 0.1
        parts = self.parts(1)
        self.assertEqual(len(parts[SkeletonPart.HIDDEN_BODY]), 2)
        self.assertEqual(len(parts[SkeletonPart.BODY]), len(pose_landmarks.CONNECTIONS) - 2)

    def test_frames_without_a_body_or_outside_the_clip_draw_nothing(self) -> None:
        self.cache.pose.found[2] = False
        self.cache.right_hand.found[2] = False
        self.assertEqual(self.parts(2), {})
        self.assertEqual(self.parts(-1), {})
        self.assertEqual(self.parts(3), {})


if __name__ == "__main__":
    unittest.main()
