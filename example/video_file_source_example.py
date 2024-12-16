"""An example of using the VideoFileSource."""
# Copyright (C) 2023, NG:ITL

from pathlib import Path
from typing import Any
from json import load
import sys

from vehicle_tracking.image_sources import VideoFileSource
from vehicle_tracking.tracker import VehicleTracker


# Constants
CURRENT_DIR = Path(__file__).parent


if __name__ == "__main__":
    config_path = Path().cwd() / "vehicle_tracking_config.json"
    with open(config_path, "r", encoding="utf-8") as config_file:
        config: dict[str, Any] = load(config_file)["starting_tracker"]

    video_file_filepath: Path = (config_path.parent / config["video_file_path"]).resolve()

    if config["use_camera_stream"]:
        print(f"Set 'use_camera_stream' to false in '{config_path}'.")
        sys.exit(1)
    elif not config["video_file_path"]:
        print(
            "Please enter the video you want to use. (e.g. 'resources/test_video_1.h265')"
            "You can also use the download script in 'resources/download_resources.py' "
            "to download a video from the NG:ITL Cloud."
        )
        sys.exit(1)
    elif video_file_filepath.is_file():
        print(f"The path '{video_file_filepath}' is not a file. Please enter a valid file path.")
        sys.exit(1)

    source = VideoFileSource(video_file_filepath, frame_rate=60)
    tracker = VehicleTracker(source)
    while True:
        tracker.step()
