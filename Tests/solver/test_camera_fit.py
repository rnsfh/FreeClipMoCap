import unittest

import numpy as np

from bl_ext.user_default.freeclipmocap.solver.camera_fit import CameraFitError, fit_camera, project

_UP = np.array([0.0, 0.0, 1.0])
_WIDTH = 1920
_HEIGHT = 1080


def look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    back = (position - target) / np.linalg.norm(position - target)
    right = np.cross(_UP, back)
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(back, right), back))


def actor_points(generator: np.random.Generator, count: int) -> np.ndarray:
    return np.column_stack((
        generator.uniform(-0.4, 0.4, count),
        generator.uniform(-0.3, 0.3, count),
        generator.uniform(0.0, 1.8, count),
    ))


class CameraFitTests(unittest.TestCase):
    def test_recovers_an_oblique_camera_despite_noise_and_mistracked_points(self) -> None:
        generator = np.random.default_rng(3)
        world_points = actor_points(generator, 600)
        position = np.array([2.4, -3.1, 1.3])
        rotation = look_at(position, np.array([0.1, 0.0, 0.9]))
        focal = 1500.0
        image_points = project(world_points, rotation, position, focal, _WIDTH, _HEIGHT)
        image_points += generator.normal(scale=1.0, size=image_points.shape)
        mistracked = generator.choice(len(image_points), 60, replace=False)
        image_points[mistracked] += generator.uniform(-150.0, 150.0, size=(60, 2))

        fitted = fit_camera(world_points, image_points, _WIDTH, _HEIGHT, focal, _UP)

        self.assertLess(np.linalg.norm(fitted.position - position), 0.05)
        self.assertLess(np.degrees(np.arccos(np.clip((np.trace(fitted.rotation.T @ rotation) - 1) / 2, -1, 1))), 0.5)
        self.assertLess(fitted.median_error_pixels, 2.0)

    def test_finds_a_camera_behind_the_actor(self) -> None:
        generator = np.random.default_rng(4)
        world_points = actor_points(generator, 200)
        position = np.array([-0.5, 4.0, 1.0])
        rotation = look_at(position, np.array([0.0, 0.0, 0.9]))
        image_points = project(world_points, rotation, position, 1000.0, _WIDTH, _HEIGHT)

        fitted = fit_camera(world_points, image_points, _WIDTH, _HEIGHT, 1000.0, _UP)

        self.assertLess(np.linalg.norm(fitted.position - position), 0.01)

    def test_too_few_points_are_refused(self) -> None:
        world_points = actor_points(np.random.default_rng(5), 5)
        with self.assertRaises(CameraFitError):
            fit_camera(world_points, np.zeros((5, 2)), _WIDTH, _HEIGHT, 1000.0, _UP)


if __name__ == "__main__":
    unittest.main()
