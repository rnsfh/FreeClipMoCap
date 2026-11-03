from enum import StrEnum


class Role(StrEnum):
    HIPS = "Hips"
    SPINE = "Spine"
    SPINE_1 = "Spine1"
    SPINE_2 = "Spine2"
    NECK = "Neck"
    HEAD = "Head"
    LEFT_SHOULDER = "LeftShoulder"
    LEFT_ARM = "LeftArm"
    LEFT_FORE_ARM = "LeftForeArm"
    LEFT_HAND = "LeftHand"
    LEFT_UP_LEG = "LeftUpLeg"
    LEFT_LEG = "LeftLeg"
    LEFT_FOOT = "LeftFoot"
    LEFT_TOE_BASE = "LeftToeBase"
    RIGHT_SHOULDER = "RightShoulder"
    RIGHT_ARM = "RightArm"
    RIGHT_FORE_ARM = "RightForeArm"
    RIGHT_HAND = "RightHand"
    RIGHT_UP_LEG = "RightUpLeg"
    RIGHT_LEG = "RightLeg"
    RIGHT_FOOT = "RightFoot"
    RIGHT_TOE_BASE = "RightToeBase"
    LEFT_HAND_THUMB_1 = "LeftHandThumb1"
    LEFT_HAND_THUMB_2 = "LeftHandThumb2"
    LEFT_HAND_THUMB_3 = "LeftHandThumb3"
    LEFT_HAND_INDEX_1 = "LeftHandIndex1"
    LEFT_HAND_INDEX_2 = "LeftHandIndex2"
    LEFT_HAND_INDEX_3 = "LeftHandIndex3"
    LEFT_HAND_MIDDLE_1 = "LeftHandMiddle1"
    LEFT_HAND_MIDDLE_2 = "LeftHandMiddle2"
    LEFT_HAND_MIDDLE_3 = "LeftHandMiddle3"
    LEFT_HAND_RING_1 = "LeftHandRing1"
    LEFT_HAND_RING_2 = "LeftHandRing2"
    LEFT_HAND_RING_3 = "LeftHandRing3"
    LEFT_HAND_PINKY_1 = "LeftHandPinky1"
    LEFT_HAND_PINKY_2 = "LeftHandPinky2"
    LEFT_HAND_PINKY_3 = "LeftHandPinky3"
    RIGHT_HAND_THUMB_1 = "RightHandThumb1"
    RIGHT_HAND_THUMB_2 = "RightHandThumb2"
    RIGHT_HAND_THUMB_3 = "RightHandThumb3"
    RIGHT_HAND_INDEX_1 = "RightHandIndex1"
    RIGHT_HAND_INDEX_2 = "RightHandIndex2"
    RIGHT_HAND_INDEX_3 = "RightHandIndex3"
    RIGHT_HAND_MIDDLE_1 = "RightHandMiddle1"
    RIGHT_HAND_MIDDLE_2 = "RightHandMiddle2"
    RIGHT_HAND_MIDDLE_3 = "RightHandMiddle3"
    RIGHT_HAND_RING_1 = "RightHandRing1"
    RIGHT_HAND_RING_2 = "RightHandRing2"
    RIGHT_HAND_RING_3 = "RightHandRing3"
    RIGHT_HAND_PINKY_1 = "RightHandPinky1"
    RIGHT_HAND_PINKY_2 = "RightHandPinky2"
    RIGHT_HAND_PINKY_3 = "RightHandPinky3"


BODY_ROLES = (
    Role.HIPS,
    Role.SPINE,
    Role.SPINE_1,
    Role.SPINE_2,
    Role.NECK,
    Role.HEAD,
    Role.LEFT_SHOULDER,
    Role.LEFT_ARM,
    Role.LEFT_FORE_ARM,
    Role.LEFT_HAND,
    Role.LEFT_UP_LEG,
    Role.LEFT_LEG,
    Role.LEFT_FOOT,
    Role.LEFT_TOE_BASE,
    Role.RIGHT_SHOULDER,
    Role.RIGHT_ARM,
    Role.RIGHT_FORE_ARM,
    Role.RIGHT_HAND,
    Role.RIGHT_UP_LEG,
    Role.RIGHT_LEG,
    Role.RIGHT_FOOT,
    Role.RIGHT_TOE_BASE,
)

LEFT_FINGER_ROLES = (
    Role.LEFT_HAND_THUMB_1,
    Role.LEFT_HAND_THUMB_2,
    Role.LEFT_HAND_THUMB_3,
    Role.LEFT_HAND_INDEX_1,
    Role.LEFT_HAND_INDEX_2,
    Role.LEFT_HAND_INDEX_3,
    Role.LEFT_HAND_MIDDLE_1,
    Role.LEFT_HAND_MIDDLE_2,
    Role.LEFT_HAND_MIDDLE_3,
    Role.LEFT_HAND_RING_1,
    Role.LEFT_HAND_RING_2,
    Role.LEFT_HAND_RING_3,
    Role.LEFT_HAND_PINKY_1,
    Role.LEFT_HAND_PINKY_2,
    Role.LEFT_HAND_PINKY_3,
)

RIGHT_FINGER_ROLES = (
    Role.RIGHT_HAND_THUMB_1,
    Role.RIGHT_HAND_THUMB_2,
    Role.RIGHT_HAND_THUMB_3,
    Role.RIGHT_HAND_INDEX_1,
    Role.RIGHT_HAND_INDEX_2,
    Role.RIGHT_HAND_INDEX_3,
    Role.RIGHT_HAND_MIDDLE_1,
    Role.RIGHT_HAND_MIDDLE_2,
    Role.RIGHT_HAND_MIDDLE_3,
    Role.RIGHT_HAND_RING_1,
    Role.RIGHT_HAND_RING_2,
    Role.RIGHT_HAND_RING_3,
    Role.RIGHT_HAND_PINKY_1,
    Role.RIGHT_HAND_PINKY_2,
    Role.RIGHT_HAND_PINKY_3,
)

REQUIRED_ROLES = (
    Role.HIPS,
    Role.SPINE,
    Role.HEAD,
    Role.LEFT_ARM,
    Role.LEFT_FORE_ARM,
    Role.LEFT_HAND,
    Role.LEFT_UP_LEG,
    Role.LEFT_LEG,
    Role.LEFT_FOOT,
    Role.RIGHT_ARM,
    Role.RIGHT_FORE_ARM,
    Role.RIGHT_HAND,
    Role.RIGHT_UP_LEG,
    Role.RIGHT_LEG,
    Role.RIGHT_FOOT,
)
