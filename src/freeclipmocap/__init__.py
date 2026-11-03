from .ui import (
    bone_map_operators,
    capture_operators,
    export_operators,
    panel,
    properties,
    solve_operator,
    video_operators,
    video_overlay,
)

_modules = (
    video_overlay,
    properties,
    capture_operators,
    bone_map_operators,
    solve_operator,
    video_operators,
    export_operators,
    panel,
)


def register() -> None:
    for module in _modules:
        module.register()


def unregister() -> None:
    for module in reversed(_modules):
        module.unregister()
