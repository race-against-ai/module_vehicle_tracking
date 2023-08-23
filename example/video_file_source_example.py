# Copyright (C) 2023 NG:ITL
from vehicle_tracking.image_sources import VideoFileSource
from vehicle_tracking.tracker import VehicleTracker
from pathlib import Path
from typing import Any
from json import load
import sys


# Constants
CURRENT_DIR = Path(__file__).parent
VIDEO_FILE_SOURCE_FILEPATH = CURRENT_DIR.parent / "resources/test_video_1.h265"


if __name__ == "__main__":
    config_path = Path().cwd() / "vehicle_tracking_config.json"
    with open(config_path, "r") as config_file:
        config: dict[str, Any] = load(config_file)["starting_tracker"]
    if not config_path.parent / "video_file_path":
        print("Please run download script in 'resources/download_resources.py'")
        sys.exit(-1)

    source = VideoFileSource(VIDEO_FILE_SOURCE_FILEPATH, frame_rate=60)
    tracker = VehicleTracker(source)
    while True:
        tracker.step()
