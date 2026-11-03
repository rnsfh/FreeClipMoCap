import numpy as np

_EPSILON = 1e-9


def normalized(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(lengths, _EPSILON)


def frame_from_directions(direction: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_axis = normalized(direction)
    z_unnormalized = reference - np.sum(reference * y_axis, axis=-1, keepdims=True) * y_axis
    z_lengths = np.linalg.norm(z_unnormalized, axis=-1)
    z_axis = z_unnormalized / np.maximum(z_lengths, _EPSILON)[..., None]
    x_axis = np.cross(y_axis, z_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=-1), z_lengths


def rotation_between(from_vectors: np.ndarray, to_vectors: np.ndarray) -> np.ndarray:
    from_unit = normalized(from_vectors)
    to_unit = normalized(to_vectors)
    cosine = np.sum(from_unit * to_unit, axis=-1)
    axis = np.cross(from_unit, to_unit)
    quaternions = np.concatenate(((1.0 + cosine)[..., None], axis), axis=-1)
    opposite = cosine < -1.0 + 1e-6
    if np.any(opposite):
        helper = np.where(np.abs(from_unit[..., :1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        perpendicular = normalized(np.cross(from_unit, helper))
        quaternions[opposite] = np.concatenate((np.zeros_like(cosine)[..., None], perpendicular), axis=-1)[opposite]
    return matrices_from_quaternions(normalized(quaternions))


def matrices_from_quaternions(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(normalized(quaternions), -1, 0)
    return np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), axis=-1),
            np.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), axis=-1),
            np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), axis=-1),
        ),
        axis=-2,
    )


def quaternions_from_matrices(matrices: np.ndarray) -> np.ndarray:
    m = matrices
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    candidates = np.stack(
        (
            np.stack((1 + trace, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]), -1),
            np.stack((m[..., 2, 1] - m[..., 1, 2], 1 + 2 * m[..., 0, 0] - trace, m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]), -1),
            np.stack((m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], 1 + 2 * m[..., 1, 1] - trace, m[..., 1, 2] + m[..., 2, 1]), -1),
            np.stack((m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 + 2 * m[..., 2, 2] - trace), -1),
        ),
        axis=-2,
    )
    largest = np.argmax(np.stack((trace, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]), -1), axis=-1)
    chosen = np.take_along_axis(candidates, largest[..., None, None], axis=-2)[..., 0, :]
    quaternions = normalized(chosen)
    return quaternions * np.where(quaternions[..., :1] < 0, -1.0, 1.0)


def slerp(start: np.ndarray, end: np.ndarray, fraction: np.ndarray | float) -> np.ndarray:
    fraction = np.asarray(fraction, dtype=np.float64)
    if fraction.ndim > 0:
        fraction = fraction[..., None]
    dot = np.sum(start * end, axis=-1, keepdims=True)
    end = np.where(dot < 0, -end, end)
    dot = np.abs(dot)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    sine = np.sin(angle)
    is_small = sine < 1e-6
    safe_sine = np.where(is_small, 1.0, sine)
    start_weight = np.where(is_small, 1.0 - fraction, np.sin((1.0 - fraction) * angle) / safe_sine)
    end_weight = np.where(is_small, fraction, np.sin(fraction * angle) / safe_sine)
    return normalized(start_weight * start + end_weight * end)


def slerp_matrices(start: np.ndarray, end: np.ndarray, fraction: np.ndarray | float) -> np.ndarray:
    return matrices_from_quaternions(slerp(quaternions_from_matrices(start), quaternions_from_matrices(end), fraction))


def make_continuous(quaternions: np.ndarray) -> np.ndarray:
    continuous = quaternions.copy()
    for frame_index in range(1, len(continuous)):
        if np.dot(continuous[frame_index], continuous[frame_index - 1]) < 0:
            continuous[frame_index] = -continuous[frame_index]
    return continuous


def log_matrices(matrices: np.ndarray) -> np.ndarray:
    quaternions = quaternions_from_matrices(matrices)
    vector_part = quaternions[..., 1:]
    sine = np.linalg.norm(vector_part, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sine, quaternions[..., :1])
    return vector_part * np.where(sine > _EPSILON, angle / np.maximum(sine, _EPSILON), 2.0)


def exp_vectors(vectors: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.where(angle > _EPSILON, np.sin(angle / 2.0) / np.maximum(angle, _EPSILON), 0.5)
    return matrices_from_quaternions(np.concatenate((np.cos(angle / 2.0), vectors * scale), axis=-1))
