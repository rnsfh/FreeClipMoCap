from pathlib import Path

import bpy

_FIT_DELAY_SECONDS = 0.2
_FIRST_VIDEO_SCENE_FRAME = 1
_SAME_FRAME_RATE_SHARE = 1e-4


def load_video_clip(
    video_path: Path, started_later_seconds: float = 0.0, first_camera_frame_rate: float | None = None
) -> bpy.types.MovieClip:
    clip = bpy.data.movieclips.load(str(video_path), check_existing=True)
    if clip.animation_data is not None:
        clip.driver_remove("frame_start")
    frame_rate = clip.fps
    if first_camera_frame_rate is None or abs(frame_rate / first_camera_frame_rate - 1.0) < _SAME_FRAME_RATE_SHARE:
        clip.frame_start = _FIRST_VIDEO_SCENE_FRAME + int(round(started_later_seconds * frame_rate))
        return clip
    # Blender shows one clip frame per scene frame, so a video filmed at another frame rate would drift away from
    # camera 1. At scene frame n (camera 1's video frame n) the clip must show its frame filmed at that moment:
    # round(((n - first) / first camera rate - started later) * its rate) + 1, which a start frame of n minus that,
    # plus 1, gives. A simple expression, so it runs without Blender's Python auto-run.
    driver = clip.driver_add("frame_start").driver
    driver.type = "SCRIPTED"
    driver.expression = (
        f"frame - round((frame - {_FIRST_VIDEO_SCENE_FRAME}) * {frame_rate / first_camera_frame_rate:.9f} "
        f"- {started_later_seconds * frame_rate:.6f})"
    )
    return clip


def load_take_clips(properties: bpy.types.PropertyGroup) -> list[bpy.types.MovieClip | None]:
    clips: list[bpy.types.MovieClip | None] = []
    first_camera_frame_rate: float | None = None
    for camera_index, (video_path, started_later_seconds) in enumerate(
        zip(take_video_paths(properties), take_start_delays_seconds(properties))
    ):
        if not video_path.is_file():
            clips.append(None)
            continue
        clip = load_video_clip(video_path, started_later_seconds, None if camera_index == 0 else first_camera_frame_rate)
        if camera_index == 0:
            first_camera_frame_rate = clip.fps
        clips.append(clip)
    return clips


def take_video_paths(properties: bpy.types.PropertyGroup) -> list[Path]:
    paths = [properties.video_path] + [camera.video_path for camera in properties.extra_videos]
    return [Path(bpy.path.abspath(path)) for path in paths]


def take_start_delays_seconds(properties: bpy.types.PropertyGroup) -> list[float]:
    return [0.0] + [camera.started_later_seconds for camera in properties.extra_videos]


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


def _split_off_clip_editor(context: bpy.types.Context, area: bpy.types.Area, direction: str) -> bpy.types.Area:
    screen = context.screen
    pointers_before = {existing.as_pointer() for existing in screen.areas}
    with context.temp_override(area=area, region=_window_region(area)):
        bpy.ops.screen.area_split(direction=direction, factor=0.5)
    new_area = next(existing for existing in screen.areas if existing.as_pointer() not in pointers_before)
    new_area.type = "CLIP_EDITOR"
    return new_area


def _clip_editor_areas_top_first(screen: bpy.types.Screen) -> list[bpy.types.Area]:
    return sorted((area for area in screen.areas if area.type == "CLIP_EDITOR"), key=lambda area: (-area.y, area.x))


def _clip_editor_areas(context: bpy.types.Context, count: int) -> list[bpy.types.Area]:
    screen = context.screen
    areas = _clip_editor_areas_top_first(screen)
    if not areas:
        viewports = [area for area in screen.areas if area.type == "VIEW_3D"]
        if not viewports:
            return []
        _split_off_clip_editor(context, max(viewports, key=lambda area: area.width * area.height), "VERTICAL")
        areas = _clip_editor_areas_top_first(screen)
    while len(areas) < count:
        _split_off_clip_editor(context, max(areas, key=lambda area: area.height), "HORIZONTAL")
        areas = _clip_editor_areas_top_first(screen)
    return areas[:count]


class ShowVideoOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.show_video"
    bl_label = "Show Video"
    bl_description = (
        "Show the video next to the 3D viewport, synced to the take (splits the largest 3D viewport the first time). "
        "With several cameras, each camera's video is shown in its own editor, stacked, kept in step with camera 1 by "
        "its Sync offset and frame rate"
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return bool(context.scene.freeclipmocap.video_path) and context.screen is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        scene = context.scene
        properties = scene.freeclipmocap
        first_video_path = take_video_paths(properties)[0]
        if not first_video_path.is_file():
            self.report({"ERROR"}, f"Video not found: {first_video_path}")
            return {"CANCELLED"}
        if not any(area.type in {"VIEW_3D", "CLIP_EDITOR"} for area in context.screen.areas):
            self.report({"ERROR"}, "Open a 3D viewport first; the video is shown next to it.")
            return {"CANCELLED"}
        take_clips = load_take_clips(properties)
        missing = [camera_index for camera_index, clip in enumerate(take_clips, start=1) if clip is None]
        clips = [clip for clip in take_clips if clip is not None]
        areas = _clip_editor_areas(context, len(clips))
        window = context.window
        for area, clip in zip(areas, clips):
            space = area.spaces.active
            space.clip = clip
            space.view = "CLIP"
            space.show_region_toolbar = False
            space.show_region_ui = False
            bpy.app.timers.register(lambda area=area: _fit_video(window, area), first_interval=_FIT_DELAY_SECONDS)
        if properties.video_end_frame == 0:
            properties.video_end_frame = areas[0].spaces.active.clip.frame_duration
        scene.sync_mode = "FRAME_DROP"
        context.screen.use_play_clip_editors = True
        shown = "Video shown" if len(clips) == 1 else f"Videos of {len(clips)} cameras shown, in step with camera 1,"
        message = f"{shown} next to the viewport; redrawn during playback, which drops frames to stay in real time."
        if missing:
            message += f" Not found, so not shown: camera {', '.join(str(camera_index) for camera_index in missing)}."
        self.report({"WARNING"} if missing else {"INFO"}, message)
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
