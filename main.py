# Copyright (C) 2022 NG:ITL
from vehicle_tracking.image_sources import VideoFileSource, CameraStreamSource
from vehicle_tracking.tracker import VehicleTracker


# Constants
FRAME_RECEIVE_LINK = "ipc:///tmp/RAAI/camera_frame.ipc"


if __name__ == "__main__":
    source = CameraStreamSource(FRAME_RECEIVE_LINK)
    tracker = VehicleTracker(source)
    while True:
        tracker.step()
