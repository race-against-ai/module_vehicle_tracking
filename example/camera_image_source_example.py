# Copyright (C) 2023, NG:ITL
from vehicle_tracking.image_sources import CameraStreamSource
from vehicle_tracking.tracker import VehicleTracker
from pathlib import Path
from typing import Any
from json import load


if __name__ == "__main__":
    config_file_path = Path().cwd() / "vehicle_tracking_config.json"
    with open(config_file_path, "r") as f:
        config: dict[str, Any] = load(f)["pynng"]["subscribers"]
        camera_frame_link: str = config["camera_frame_receiver"]["address"]

    source = CameraStreamSource(camera_frame_link)
    tracker = VehicleTracker(source)

    while True:
        tracker.step()
