import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum


class PoseModelVariant(StrEnum):
    LITE = "lite"
    FULL = "full"
    HEAVY = "heavy"


class HandSearch(StrEnum):
    FULL_FRAME = "full-frame"
    WRIST_CROPS = "wrist-crops"


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    pose_model_variant: PoseModelVariant
    hand_search: HandSearch = HandSearch.WRIST_CROPS

    def to_json(self) -> str:
        values = {"pose_model_variant": str(self.pose_model_variant)}
        if self.hand_search != HandSearch.FULL_FRAME:
            values["hand_search"] = str(self.hand_search)
        return json.dumps(values, sort_keys=True)

    def short_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:8]

    @classmethod
    def from_json(cls, text: str) -> "CaptureSettings":
        values = json.loads(text)
        return cls(
            pose_model_variant=PoseModelVariant(values["pose_model_variant"]),
            hand_search=HandSearch(values.get("hand_search", HandSearch.FULL_FRAME)),
        )
