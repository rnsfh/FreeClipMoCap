from pathlib import Path

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from ..capture.landmark_cache import CacheFormatError, LandmarkCache, cache_path_for
from ..capture.tracked_skeleton import SkeletonPart, tracked_skeleton_lines
from ..solver.mediapipe_table import MEDIAPIPE_TABLE

_LINE_WIDTH_PIXELS = 2.5
_COLORS = {
    SkeletonPart.BODY: (0.35, 0.95, 0.35, 0.95),
    SkeletonPart.HIDDEN_BODY: (0.6, 0.6, 0.6, 0.45),
    SkeletonPart.LEFT_HAND: (1.0, 0.62, 0.15, 0.95),
    SkeletonPart.RIGHT_HAND: (0.25, 0.55, 1.0, 0.95),
}


class _LoadedCache:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.modified_time_nanoseconds = 0
        self.cache: LandmarkCache | None = None

    def get(self, path: Path) -> LandmarkCache | None:
        try:
            modified_time_nanoseconds = path.stat().st_mtime_ns
        except OSError:
            return None
        if path != self.path or modified_time_nanoseconds != self.modified_time_nanoseconds:
            self.path = path
            self.modified_time_nanoseconds = modified_time_nanoseconds
            try:
                self.cache = LandmarkCache.load(path)
            except (OSError, ValueError, KeyError, CacheFormatError):
                self.cache = None
        return self.cache


_loaded_cache = _LoadedCache()
_draw_handler: object | None = None


def _same_file(first: Path, second: Path) -> bool:
    try:
        return first.resolve() == second.resolve()
    except OSError:
        return False


def _draw_tracked_skeleton() -> None:
    context = bpy.context
    space = context.space_data
    region = context.region
    scene = context.scene
    properties = scene.freeclipmocap
    clip = getattr(space, "clip", None)
    if not properties.show_tracked_skeleton or clip is None or not properties.video_path:
        return
    video_path = Path(bpy.path.abspath(properties.video_path))
    if not _same_file(Path(bpy.path.abspath(clip.filepath)), video_path):
        return
    cache = _loaded_cache.get(cache_path_for(video_path, properties.capture_settings()))
    if cache is None:
        return
    frame_index = scene.frame_current - clip.frame_start
    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("lineWidth", _LINE_WIDTH_PIXELS)
    gpu.state.blend_set("ALPHA")
    for skeleton_lines in tracked_skeleton_lines(cache, frame_index, MEDIAPIPE_TABLE.minimum_visibility):
        normalized = skeleton_lines.segments.reshape(-1, 2)
        positions = [
            (*region.view2d.view_to_region(float(x), 1.0 - float(y), clip=False), 0.0)
            for x, y in normalized
            if np.isfinite(x) and np.isfinite(y)
        ]
        if len(positions) < 2:
            continue
        shader.uniform_float("color", _COLORS[skeleton_lines.part])
        batch_for_shader(shader, "LINES", {"pos": positions}).draw(shader)
    gpu.state.blend_set("NONE")


def redraw_clip_editors(context: bpy.types.Context) -> None:
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "CLIP_EDITOR":
                area.tag_redraw()


def register() -> None:
    global _draw_handler
    _draw_handler = bpy.types.SpaceClipEditor.draw_handler_add(_draw_tracked_skeleton, (), "WINDOW", "POST_PIXEL")


def unregister() -> None:
    global _draw_handler
    if _draw_handler is not None:
        bpy.types.SpaceClipEditor.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None
