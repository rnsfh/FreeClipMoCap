import numpy as np

from bl_ext.user_default.freeclipmocap.capture.capture_settings import CaptureSettings, PoseModelVariant
from bl_ext.user_default.freeclipmocap.capture.landmark_cache import HandTrack, LandmarkCache, PoseTrack, VideoSource
from bl_ext.user_default.freeclipmocap.constants import hand_landmarks, pose_landmarks, tracking_models
from bl_ext.user_default.freeclipmocap.constants.bone_roles import Role
from bl_ext.user_default.freeclipmocap.rig.bone_map import BoneMap
from bl_ext.user_default.freeclipmocap.solver.rig_rest import RigRest

_FINGER_ROWS = {"Index": -0.03, "Middle": -0.01, "Ring": 0.01, "Pinky": 0.03}


def _bone_table() -> list[tuple[str, str | None, tuple[float, float, float], tuple[float, float, float]]]:
    bones: list[tuple[str, str | None, tuple[float, float, float], tuple[float, float, float]]] = [
        ("Hips", None, (0.0, 0.0, 0.95), (0.0, 0.0, 1.1)),
        ("Spine", "Hips", (0.0, 0.0, 1.1), (0.0, 0.0, 1.2)),
        ("Spine1", "Spine", (0.0, 0.0, 1.2), (0.0, 0.0, 1.3)),
        ("Spine2", "Spine1", (0.0, 0.0, 1.3), (0.0, 0.0, 1.42)),
        ("Neck", "Spine2", (0.0, 0.0, 1.42), (0.0, 0.0, 1.55)),
        ("Head", "Neck", (0.0, 0.0, 1.55), (0.0, 0.0, 1.75)),
        ("HeadTop_End", "Head", (0.0, 0.0, 1.75), (0.0, 0.0, 1.85)),
    ]
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        def point(x: float, y: float, z: float) -> tuple[float, float, float]:
            return (sign * x, y, z)

        bones += [
            (f"{side}Shoulder", "Spine2", point(0.05, 0.0, 1.42), point(0.18, 0.0, 1.42)),
            (f"{side}Arm", f"{side}Shoulder", point(0.18, 0.0, 1.42), point(0.46, 0.0, 1.42)),
            (f"{side}ForeArm", f"{side}Arm", point(0.46, 0.0, 1.42), point(0.72, 0.0, 1.42)),
            (f"{side}Hand", f"{side}ForeArm", point(0.72, 0.0, 1.42), point(0.80, 0.0, 1.42)),
            (f"{side}UpLeg", "Hips", point(0.1, 0.0, 0.95), point(0.1, 0.0, 0.5)),
            (f"{side}Leg", f"{side}UpLeg", point(0.1, 0.0, 0.5), point(0.1, 0.0, 0.08)),
            (f"{side}Foot", f"{side}Leg", point(0.1, 0.0, 0.08), point(0.1, -0.12, 0.02)),
            (f"{side}ToeBase", f"{side}Foot", point(0.1, -0.12, 0.02), point(0.1, -0.18, 0.02)),
        ]
        for finger, row in _FINGER_ROWS.items():
            parent = f"{side}Hand"
            for joint in range(1, 5):
                x = 0.81 + 0.03 * (joint - 1)
                bones.append((f"{side}Hand{finger}{joint}", parent, point(x, row, 1.42), point(x + 0.03, row, 1.42)))
                parent = f"{side}Hand{finger}{joint}"
        parent = f"{side}Hand"
        for joint in range(1, 5):
            start = 0.75 + 0.021 * (joint - 1)
            bones.append((
                f"{side}HandThumb{joint}",
                parent,
                point(start, -0.04 - 0.021 * (joint - 1), 1.41),
                point(start + 0.021, -0.04 - 0.021 * joint, 1.41),
            ))
            parent = f"{side}HandThumb{joint}"
    return bones


def _rest_rotation(direction: np.ndarray, roll_degrees: float) -> np.ndarray:
    y_axis = direction / np.linalg.norm(direction)
    reference = np.array([0.0, -1.0, 0.0]) if abs(y_axis[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    z_axis = reference - np.dot(reference, y_axis) * y_axis
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    angle = np.radians(roll_degrees)
    rolled_x = np.cos(angle) * x_axis + np.sin(angle) * z_axis
    rolled_z = np.cross(rolled_x, y_axis)
    return np.stack((rolled_x, y_axis, rolled_z), axis=-1)


def make_rig(prefix: str = "mixamorig:") -> tuple[RigRest, BoneMap]:
    bones = _bone_table()
    roll_generator = np.random.default_rng(7)
    rest_rotations = np.array([
        _rest_rotation(np.subtract(tail, head), roll_generator.uniform(-180.0, 180.0)) for _name, _parent, head, tail in bones
    ])
    names = [f"{prefix}{name}" for name, _parent, _head, _tail in bones]
    parent_indices = tuple(-1 if parent is None else names.index(f"{prefix}{parent}") for _name, parent, _head, _tail in bones)
    rig = RigRest(
        bone_names=tuple(names),
        parent_indices=parent_indices,
        rest_rotations=rest_rotations,
        heads=np.array([head for _name, _parent, head, _tail in bones], dtype=np.float64),
        tails=np.array([tail for _name, _parent, _head, tail in bones], dtype=np.float64),
    )
    bone_map = BoneMap(bones_by_role={role: f"{prefix}{role.value}" for role in Role if f"{prefix}{role.value}" in names})
    return rig, bone_map


def rest_landmarks(rig: RigRest) -> dict[str, np.ndarray]:
    def head(name: str) -> np.ndarray:
        return rig.heads[rig.index_of(f"mixamorig:{name}")]

    def tail(name: str) -> np.ndarray:
        return rig.tails[rig.index_of(f"mixamorig:{name}")]

    pose = np.zeros((pose_landmarks.LANDMARK_COUNT, 3))
    pose[pose_landmarks.LEFT_EAR] = (0.08, 0.0, 1.65)
    pose[pose_landmarks.RIGHT_EAR] = (-0.08, 0.0, 1.65)
    pose[pose_landmarks.LEFT_EYE] = (0.03, -0.09, 1.65 + 0.09 * np.tan(np.radians(10.0)))
    pose[pose_landmarks.RIGHT_EYE] = (-0.03, -0.09, 1.65 + 0.09 * np.tan(np.radians(10.0)))
    pose[pose_landmarks.NOSE] = (0.0, -0.11, 1.6)
    hands = {}
    for side in ("Left", "Right"):
        upper = side.upper()
        for landmark_name, bone_name in (
            ("SHOULDER", "Arm"), ("ELBOW", "ForeArm"), ("WRIST", "Hand"), ("HIP", "UpLeg"), ("KNEE", "Leg"),
            ("ANKLE", "Foot"), ("FOOT_INDEX", "ToeBase"), ("INDEX", "HandIndex1"), ("PINKY", "HandPinky1"),
            ("THUMB", "HandThumb2"),
        ):
            pose[getattr(pose_landmarks, f"{upper}_{landmark_name}")] = head(f"{side}{bone_name}")
        pose[getattr(pose_landmarks, f"{upper}_HEEL")] = head(f"{side}Foot") + (0.0, 0.03, -0.06)
        hand = np.zeros((hand_landmarks.LANDMARK_COUNT, 3))
        hand[hand_landmarks.WRIST] = head(f"{side}Hand")
        for finger, first_index in (("Thumb", 1), ("Index", 5), ("Middle", 9), ("Ring", 13), ("Pinky", 17)):
            for joint in range(3):
                hand[first_index + joint] = head(f"{side}Hand{finger}{joint + 1}")
            hand[first_index + 3] = tail(f"{side}Hand{finger}3")
        hands[side] = hand
    return {"pose": pose, "Left": hands["Left"], "Right": hands["Right"]}


def to_mediapipe(armature_points: np.ndarray) -> np.ndarray:
    return np.stack((armature_points[..., 0], -armature_points[..., 2], armature_points[..., 1]), axis=-1)


def to_image(armature_points: np.ndarray) -> np.ndarray:
    return np.stack((0.5 + armature_points[..., 0] / 4.0, 0.95 - armature_points[..., 2] / 2.2, armature_points[..., 1]), axis=-1)


def rotation_about(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    angle = np.radians(degrees)
    cross_matrix = np.array([[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]])
    return np.eye(3) + np.sin(angle) * cross_matrix + (1 - np.cos(angle)) * cross_matrix @ cross_matrix


def make_cache(
    frames: list[dict[str, np.ndarray]],
    left_hand_found: bool = True,
    pose_visibility: np.ndarray | None = None,
    tracking_model: str = tracking_models.MEDIAPIPE_POSE_HANDS,
) -> LandmarkCache:
    frame_count = len(frames)

    def hand_track(side: str, found: bool) -> HandTrack:
        world = np.stack([to_mediapipe(frame[side]) for frame in frames]).astype(np.float32)
        if not found:
            world[:] = np.nan
        return HandTrack(
            found=np.full(frame_count, found),
            image_landmarks=np.zeros_like(world),
            world_landmarks=world,
            score=np.ones(frame_count, dtype=np.float32),
        )

    pose_world = np.stack([to_mediapipe(frame["pose"]) for frame in frames]).astype(np.float32)
    return LandmarkCache(
        tracking_model=tracking_model,
        settings=CaptureSettings(pose_model_variant=PoseModelVariant.HEAVY),
        video=VideoSource(file_size=1, modified_time_nanoseconds=1, frame_rate=30.0, width=640, height=480),
        frame_indices=np.arange(frame_count, dtype=np.int32),
        timestamps_milliseconds=np.round(np.arange(frame_count) * 1000.0 / 30.0).astype(np.int64),
        pose=PoseTrack(
            found=np.ones(frame_count, dtype=bool),
            image_landmarks=np.stack([to_image(frame["pose"]) for frame in frames]).astype(np.float32),
            world_landmarks=pose_world,
            visibility=np.ones(pose_world.shape[:2], dtype=np.float32) if pose_visibility is None else pose_visibility,
            presence=np.ones(pose_world.shape[:2], dtype=np.float32),
        ),
        left_hand=hand_track("Left", left_hand_found),
        right_hand=hand_track("Right", True),
    )
