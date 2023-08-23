# Copyright (C) 2023, NG:ITL
from vehicle_tracking.image_sources import CameraStreamSource, VideoFileSource
from vehicle_tracking.tracker import VehicleTracker
from pathlib import Path
from typing import Any
from json import load


if __name__ == "__main__":
    config_path = Path().cwd() / "vehicle_tracking_config.json"
    with open(config_path, "r") as config_file:
        config: dict[str, Any] = load(config_file)
        starting_tracker_config = config["starting_tracker"]
        pynng_subscriber_config = config["pynng"]["subscribers"]

    if starting_tracker_config["use_camera_stream"]:
        source = CameraStreamSource(pynng_subscriber_config["camera_frame_receiver"]["address"])
    else:
        source = VideoFileSource(config_path.parent / starting_tracker_config["video_file_path"], 1000)
    tracker = VehicleTracker(source)
    print("x")
    while True:
        tracker.step()
