# Copyright (C) 2023, NG:ITL
from vehicle_tracking.image_sources import CameraStreamSource, VideoFileSource
from vehicle_tracking.tracker import VehicleTracker
from pathlib import Path


# Constants
FRAME_RECEIVE_LINK = "ipc:///tmp/RAAI/camera_frame.ipc"


if __name__ == "__main__":
    # source = CameraStreamSource(FRAME_RECEIVE_LINK)
    source = VideoFileSource(Path("resources/test_video_1.h265"), 1000)
    tracker = VehicleTracker(source)
    while True:
        tracker.step()
