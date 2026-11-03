import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..constants.bone_roles import Role
from .bone_map import BoneMap

_FORMAT_VERSION = 1
_VALID_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _\-]*")


class PresetError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class BoneMapPreset:
    name: str
    bones_by_role: Mapping[Role, str]
    is_built_in: bool
    rig_settings: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    source_path: Path | None = field(default=None, compare=False)

    def matched_to(self, bone_names: Iterable[str]) -> BoneMap:
        bone_names = list(bone_names)
        exact_names = set(bone_names)
        bones_by_short_name: dict[str, list[str]] = {}
        for bone_name in bone_names:
            bones_by_short_name.setdefault(without_namespace(bone_name), []).append(bone_name)
        matched: dict[Role, str] = {}
        for role, preset_bone_name in self.bones_by_role.items():
            if preset_bone_name in exact_names:
                matched[role] = preset_bone_name
                continue
            candidates = bones_by_short_name.get(without_namespace(preset_bone_name), [])
            if len(candidates) == 1:
                matched[role] = candidates[0]
        return BoneMap(bones_by_role=matched)

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = preset_path(folder, self.name)
        document = {
            "format_version": _FORMAT_VERSION,
            "name": self.name,
            "bones_by_role": {str(role): bone_name for role, bone_name in self.bones_by_role.items()},
        }
        if self.rig_settings:
            document["rig_settings"] = {bone_name: dict(values) for bone_name, values in self.rig_settings.items()}
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path, is_built_in: bool) -> "BoneMapPreset":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PresetError(f"Cannot read the bone map preset {path}: {error}") from error
        if document.get("format_version") != _FORMAT_VERSION:
            raise PresetError(f"{path} has format version {document.get('format_version')}, expected {_FORMAT_VERSION}.")
        bones_by_role: dict[Role, str] = {}
        for role_name, bone_name in document.get("bones_by_role", {}).items():
            try:
                bones_by_role[Role(role_name)] = str(bone_name)
            except ValueError as error:
                raise PresetError(f"{path} names an unknown role: {role_name}.") from error
        return cls(
            name=str(document.get("name", path.stem)),
            bones_by_role=bones_by_role,
            is_built_in=is_built_in,
            rig_settings=read_rig_settings(document.get("rig_settings", {}), str(path)),
            source_path=path,
        )

    def delete(self) -> None:
        if self.is_built_in or self.source_path is None:
            raise PresetError(f"The preset {self.name} is built in and cannot be deleted.")
        self.source_path.unlink()


def read_rig_settings(document: object, source: str) -> dict[str, dict[str, float]]:
    if not isinstance(document, dict) or not all(isinstance(values, dict) for values in document.values()):
        raise PresetError(f"{source}: rig_settings must map bone names to {{setting: number}}.")
    try:
        return {
            str(bone_name): {str(setting): float(value) for setting, value in values.items()}
            for bone_name, values in document.items()
        }
    except (TypeError, ValueError) as error:
        raise PresetError(f"{source}: rig_settings values must be numbers.") from error


def without_namespace(bone_name: str) -> str:
    return bone_name.rsplit(":", 1)[-1]


def is_valid_preset_name(name: str) -> bool:
    return _VALID_NAME_PATTERN.fullmatch(name) is not None


def preset_path(folder: Path, name: str) -> Path:
    if not is_valid_preset_name(name):
        raise PresetError(f"Preset names may use letters, digits, spaces, '-' and '_' only: {name!r}.")
    return folder / f"{name}.json"


def load_presets(built_in_folder: Path, user_folder: Path | None) -> tuple[list[BoneMapPreset], list[str]]:
    presets: list[BoneMapPreset] = []
    problems: list[str] = []
    folders = [(built_in_folder, True)] + ([(user_folder, False)] if user_folder is not None else [])
    for folder, is_built_in in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            try:
                presets.append(BoneMapPreset.load(path, is_built_in))
            except PresetError as error:
                problems.append(str(error))
    return presets, problems
