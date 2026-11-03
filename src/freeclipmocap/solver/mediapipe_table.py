from ..constants import hand_landmarks, pose_landmarks, tracking_models
from ..constants.bone_roles import Role
from .tracking_table import (
    Direction,
    FootRule,
    HandRule,
    JointPoint,
    LandmarkPoint,
    PointSet,
    RigAxis,
    RigPoint,
    RoleRule,
    SolveMode,
    TrackingTable,
)

_FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
_FINGER_LANDMARKS = {
    "Thumb": (hand_landmarks.THUMB_CMC, hand_landmarks.THUMB_MCP, hand_landmarks.THUMB_IP, hand_landmarks.THUMB_TIP),
    "Index": (hand_landmarks.INDEX_MCP, hand_landmarks.INDEX_PIP, hand_landmarks.INDEX_DIP, hand_landmarks.INDEX_TIP),
    "Middle": (hand_landmarks.MIDDLE_MCP, hand_landmarks.MIDDLE_PIP, hand_landmarks.MIDDLE_DIP, hand_landmarks.MIDDLE_TIP),
    "Ring": (hand_landmarks.RING_MCP, hand_landmarks.RING_PIP, hand_landmarks.RING_DIP, hand_landmarks.RING_TIP),
    "Pinky": (hand_landmarks.PINKY_MCP, hand_landmarks.PINKY_PIP, hand_landmarks.PINKY_DIP, hand_landmarks.PINKY_TIP),
}


def _pose(*indices: int) -> LandmarkPoint:
    return LandmarkPoint(PointSet.POSE, indices)


def _rig(*roles: Role) -> RigPoint:
    return RigPoint(roles)


_HIP_CENTER = _pose(pose_landmarks.LEFT_HIP, pose_landmarks.RIGHT_HIP)
_SHOULDER_CENTER = _pose(pose_landmarks.LEFT_SHOULDER, pose_landmarks.RIGHT_SHOULDER)
_EAR_CENTER = _pose(pose_landmarks.LEFT_EAR, pose_landmarks.RIGHT_EAR)
_EYE_CENTER = _pose(pose_landmarks.LEFT_EYE, pose_landmarks.RIGHT_EYE)
_RIG_HIP_CENTER = _rig(Role.LEFT_UP_LEG, Role.RIGHT_UP_LEG)
_RIG_SHOULDER_CENTER = _rig(Role.LEFT_ARM, Role.RIGHT_ARM)
_TORSO_UP = Direction(_HIP_CENTER, _SHOULDER_CENTER, _RIG_HIP_CENTER, _RIG_SHOULDER_CENTER)


def _side_rules(side: str) -> tuple[RoleRule, ...]:
    def landmark(name: str) -> int:
        return getattr(pose_landmarks, f"{side.upper()}_{name}")

    def role(name: str) -> Role:
        return Role(f"{side}{name}")

    shoulder = _pose(landmark("SHOULDER"))
    elbow = _pose(landmark("ELBOW"))
    wrist = _pose(landmark("WRIST"))
    index = _pose(landmark("INDEX"))
    pinky = _pose(landmark("PINKY"))
    hip = _pose(landmark("HIP"))
    knee = _pose(landmark("KNEE"))
    ankle = _pose(landmark("ANKLE"))
    foot_index = _pose(landmark("FOOT_INDEX"))
    return (
        RoleRule(role("Shoulder"), SolveMode.SWING, Direction(_SHOULDER_CENTER, shoulder)),
        RoleRule(
            role("Arm"),
            SolveMode.BENDING_LIMB,
            Direction(shoulder, elbow),
            Direction(elbow, wrist, rig_axis=RigAxis.FRONT),
        ),
        RoleRule(role("ForeArm"), SolveMode.SWING, Direction(elbow, wrist)),
        RoleRule(
            role("Hand"),
            SolveMode.FRAME,
            Direction(
                wrist,
                _pose(landmark("INDEX"), landmark("PINKY")),
                _rig(role("Hand")),
                _rig(role("HandIndex1"), role("HandPinky1")),
            ),
            Direction(index, pinky, _rig(role("HandIndex1")), _rig(role("HandPinky1"))),
        ),
        RoleRule(
            role("UpLeg"),
            SolveMode.BENDING_LIMB,
            Direction(hip, knee),
            Direction(knee, ankle, rig_axis=RigAxis.BACK),
        ),
        RoleRule(role("Leg"), SolveMode.SWING, Direction(knee, ankle)),
        RoleRule(role("Foot"), SolveMode.SWING, Direction(ankle, foot_index)),
    )


def _side_joint_points(side: str) -> tuple[JointPoint, ...]:
    return tuple(
        JointPoint(Role(f"{side}{bone_name}"), _pose(getattr(pose_landmarks, f"{side.upper()}_{landmark_name}")))
        for bone_name, landmark_name in (
            ("Arm", "SHOULDER"),
            ("ForeArm", "ELBOW"),
            ("Hand", "WRIST"),
            ("UpLeg", "HIP"),
            ("Leg", "KNEE"),
            ("Foot", "ANKLE"),
        )
    )


def _hand_rule(side: str) -> HandRule:
    point_set = PointSet.LEFT_HAND if side == "Left" else PointSet.RIGHT_HAND

    def point(index: int) -> LandmarkPoint:
        return LandmarkPoint(point_set, (index,))

    def role(name: str) -> Role:
        return Role(f"{side}{name}")

    finger_rules = tuple(
        RoleRule(
            role(f"Hand{finger}{joint}"),
            SolveMode.SWING,
            Direction(point(_FINGER_LANDMARKS[finger][joint - 1]), point(_FINGER_LANDMARKS[finger][joint])),
        )
        for finger in _FINGER_NAMES
        for joint in (1, 2, 3)
    )
    return HandRule(
        hand_role=role("Hand"),
        point_set=point_set,
        direction=Direction(
            point(hand_landmarks.WRIST),
            point(hand_landmarks.MIDDLE_MCP),
            _rig(role("Hand")),
            _rig(role("HandMiddle1")),
        ),
        twist=Direction(
            point(hand_landmarks.INDEX_MCP),
            point(hand_landmarks.PINKY_MCP),
            _rig(role("HandIndex1")),
            _rig(role("HandPinky1")),
        ),
        finger_rules=finger_rules,
    )


MEDIAPIPE_TABLE = TrackingTable(
    tracking_model=tracking_models.MEDIAPIPE_POSE_HANDS,
    minimum_visibility=0.5,
    minimum_visible_share=0.25,
    hips_rule=RoleRule(
        Role.HIPS,
        SolveMode.FRAME,
        _TORSO_UP,
        Direction(
            _pose(pose_landmarks.LEFT_HIP),
            _pose(pose_landmarks.RIGHT_HIP),
            _rig(Role.LEFT_UP_LEG),
            _rig(Role.RIGHT_UP_LEG),
        ),
    ),
    chest_direction=_TORSO_UP,
    chest_twist=Direction(
        _pose(pose_landmarks.LEFT_SHOULDER),
        _pose(pose_landmarks.RIGHT_SHOULDER),
        _rig(Role.LEFT_ARM),
        _rig(Role.RIGHT_ARM),
    ),
    spine_roles=(Role.SPINE, Role.SPINE_1, Role.SPINE_2),
    role_rules=(
        RoleRule(Role.NECK, SolveMode.SWING, Direction(_SHOULDER_CENTER, _EAR_CENTER)),
        RoleRule(
            Role.HEAD,
            SolveMode.FRAME,
            Direction(_EAR_CENTER, _EYE_CENTER, rig_axis=RigAxis.FRONT, rig_axis_tilt_up_degrees=10.0),
            Direction(
                _pose(pose_landmarks.LEFT_EAR),
                _pose(pose_landmarks.RIGHT_EAR),
                rig_axis=RigAxis.RIGHT,
            ),
        ),
        *_side_rules("Left"),
        *_side_rules("Right"),
    ),
    hand_rules=(_hand_rule("Left"), _hand_rule("Right")),
    foot_rules=(
        FootRule(
            Role.LEFT_UP_LEG,
            Role.LEFT_LEG,
            Role.LEFT_FOOT,
            Role.LEFT_TOE_BASE,
            _pose(pose_landmarks.LEFT_HEEL, pose_landmarks.LEFT_FOOT_INDEX),
        ),
        FootRule(
            Role.RIGHT_UP_LEG,
            Role.RIGHT_LEG,
            Role.RIGHT_FOOT,
            Role.RIGHT_TOE_BASE,
            _pose(pose_landmarks.RIGHT_HEEL, pose_landmarks.RIGHT_FOOT_INDEX),
        ),
    ),
    joint_points=(*_side_joint_points("Left"), *_side_joint_points("Right")),
)
