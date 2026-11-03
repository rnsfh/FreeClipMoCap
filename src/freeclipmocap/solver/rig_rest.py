from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class RigRest:
    bone_names: tuple[str, ...]
    parent_indices: tuple[int, ...]
    rest_rotations: np.ndarray
    heads: np.ndarray
    tails: np.ndarray

    def index_of(self, bone_name: str) -> int:
        return self.bone_names.index(bone_name)
