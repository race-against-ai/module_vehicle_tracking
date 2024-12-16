"""An example of using the CameraImageSource."""
# Copyright (C) 2023, NG:ITL

from pathlib import Path
from json import load

from vehicle_tracking.image_sources import CameraStreamSource
from vehicle_tracking.tracker import VehicleTracker


if __name__ == "__main__":
    config_file_path = Path().cwd() / "vehicle_tracking_config.json"
    with open(config_file_path, "r", encoding="utf-8") as f:
        camera_frame_link: str = load(f)["pynng"]["subscribers"]["camera_frame_receiver"]["address"]

    source = CameraStreamSource(camera_frame_link)
    tracker = VehicleTracker(source)

    while True:
        tracker.step()
