class CaptureCancelled(Exception):
    pass


class CaptureProgress:
    def __init__(self) -> None:
        self.frames_done = 0
        self.expected_frame_count = 0

    def set_expected_frame_count(self, frame_count: int) -> None:
        self.expected_frame_count = frame_count

    def advance(self) -> None:
        self.frames_done += 1

    @property
    def fraction(self) -> float:
        if self.expected_frame_count <= 0:
            return 0.0
        return min(self.frames_done / self.expected_frame_count, 1.0)
