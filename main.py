"""The main script calling the tracker."""
# Copyright (C) 2023, NG:ITL

from pathlib import Path
from typing import Any
from json import load

from vehicle_tracking.image_sources import CameraStreamSource, VideoFileSource
from vehicle_tracking.tracker import VehicleTracker


CONFIG_PATH = Path("./vehicle_tracking_config.json")


if __name__ == "__main__":
    if not CONFIG_PATH.exists():
        with open(Path(__file__).parent / "vehicle_tracking/templates/tracker_config.json", "r", encoding="utf-8") as template:
            with open(CONFIG_PATH, "x", encoding="utf-8") as f:
                f.write(template.read())

    with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
        config: dict[str, Any] = load(config_file)
        starting_tracker_config = config["starting_tracker"]
        pynng_subscriber_config = config["pynng"]["subscribers"]

    source: VideoFileSource | CameraStreamSource
    if starting_tracker_config["use_camera_stream"]:
        source = CameraStreamSource(pynng_subscriber_config["camera_frame_receiver"]["address"])
    else:
        source = VideoFileSource(CONFIG_PATH.parent / starting_tracker_config["video_file_path"], 1000)
    tracker = VehicleTracker(source)
    while True:
        tracker.step()
