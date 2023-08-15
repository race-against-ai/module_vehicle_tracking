# Copyright (C) 2023, NG:ITL
from vehicle_tracking.image_sources import CameraStreamSource
from vehicle_tracking.tracker import VehicleTracker


# Constants
CAMERA_IMAGE_STREAM_LINK = "icp:///ipc:///tmp/RAAI/camera_frame.ipc"


if __name__ == "__main__":
    source = CameraStreamSource(CAMERA_IMAGE_STREAM_LINK)
    tracker = VehicleTracker(source)
    while True:
        tracker.step()
