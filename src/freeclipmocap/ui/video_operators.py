from pathlib import Path

import bpy

_FIT_DELAY_SECONDS = 0.2
_FIRST_VIDEO_SCENE_FRAME = 1


def load_video_clip(video_path: Path) -> bpy.types.MovieClip:
    clip = bpy.data.movieclips.load(str(video_path), check_existing=True)
    clip.frame_start = _FIRST_VIDEO_SCENE_FRAME
    return clip


def video_frame_at(scene_frame: int) -> int:
    return scene_frame - _FIRST_VIDEO_SCENE_FRAME + 1


def scene_frame_of(video_frame: int) -> int:
    return video_frame + _FIRST_VIDEO_SCENE_FRAME - 1


def _window_region(area: bpy.types.Area) -> bpy.types.Region:
    return next(region for region in area.regions if region.type == "WINDOW")


def _fit_video(window: bpy.types.Window, area: bpy.types.Area) -> None:
    try:
        with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=_window_region(area)):
            bpy.ops.clip.view_all(fit_view=True)
    except (RuntimeError, ReferenceError, StopIteration):
        pass


def _clip_editor_area(context: bpy.types.Context) -> bpy.types.Area | None:
    screen = context.screen
    existing = next((area for area in screen.areas if area.type == "CLIP_EDITOR"), None)
    if existing is not None:
        return existing
    viewports = [area for area in screen.areas if area.type == "VIEW_3D"]
    if not viewports:
        return None
    viewport = max(viewports, key=lambda area: area.width * area.height)
    pointers_before = {area.as_pointer() for area in screen.areas}
    with context.temp_override(area=viewport, region=_window_region(viewport)):
        bpy.ops.screen.area_split(direction="VERTICAL", factor=0.5)
    new_area = next(area for area in screen.areas if area.as_pointer() not in pointers_before)
    new_area.type = "CLIP_EDITOR"
    return new_area


class ShowVideoOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.show_video"
    bl_label = "Show Video"
    bl_description = (
        "Show the video next to the 3D viewport, synced to the take (splits the largest 3D viewport the first time)"
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return bool(context.scene.freeclipmocap.video_path) and context.screen is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        scene = context.scene
        video_path = Path(bpy.path.abspath(scene.freeclipmocap.video_path))
        if not video_path.is_file():
            self.report({"ERROR"}, f"Video not found: {video_path}")
            return {"CANCELLED"}
        area = _clip_editor_area(context)
        if area is None:
            self.report({"ERROR"}, "Open a 3D viewport first; the video is shown next to it.")
            return {"CANCELLED"}
        clip = load_video_clip(video_path)
        if scene.freeclipmocap.video_end_frame == 0:
            scene.freeclipmocap.video_end_frame = clip.frame_duration
        space = area.spaces.active
        space.clip = clip
        space.view = "CLIP"
        space.show_region_toolbar = False
        space.show_region_ui = False
        window = context.window
        bpy.app.timers.register(lambda: _fit_video(window, area), first_interval=_FIT_DELAY_SECONDS)
        scene.sync_mode = "FRAME_DROP"
        context.screen.use_play_clip_editors = True
        self.report(
            {"INFO"},
            "Video shown next to the viewport; it is redrawn during playback, which drops frames to stay in real time.",
        )
        return {"FINISHED"}


class SetVideoFrameOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.set_video_frame"
    bl_label = "Set Video Frame"
    bl_options = {"REGISTER", "UNDO"}

    edge: bpy.props.EnumProperty(
        items=(
            ("START", "Start", "Use the current frame as the first video frame for Solve"),
            ("END", "End", "Use the current frame as the last video frame for Solve"),
        )
    )

    @classmethod
    def description(cls, context: bpy.types.Context, properties: bpy.types.OperatorProperties) -> str:
        return f"Use the video frame shown now as the {'first' if properties.edge == 'START' else 'last'} frame for Solve"

    def execute(self, context: bpy.types.Context) -> set[str]:
        properties = context.scene.freeclipmocap
        video_frame = video_frame_at(context.scene.frame_current)
        if video_frame < 1:
            self.report({"ERROR"}, "The video starts at frame 1; move to a frame inside the video.")
            return {"CANCELLED"}
        if self.edge == "START":
            properties.video_start_frame = video_frame
            if 0 < properties.video_end_frame < video_frame:
                properties.video_end_frame = video_frame
        else:
            properties.video_end_frame = video_frame
            properties.video_start_frame = min(properties.video_start_frame, video_frame)
        return {"FINISHED"}


register, unregister = bpy.utils.register_classes_factory((ShowVideoOperator, SetVideoFrameOperator))
