# Copyright (C) 2023 NG:ITL
import sys
from pathlib import Path

from vehicle_tracking.image_sources import VideoFileSource
from vehicle_tracking.tracker import VehicleTracker

FILE_DIR = Path(__file__).parent

# Constants
VIDEO_FILE_SOURCE_FILEPATH = FILE_DIR.parent / "resources/example_video.avi"


if __name__ == "__main__":

    if not VIDEO_FILE_SOURCE_FILEPATH.exists():
        print("Please run downloadscirpt in ...")
        sys.exit(-1)

    source = VideoFileSource(VIDEO_FILE_SOURCE_FILEPATH, frame_rate=30)
    tracker = VehicleTracker(source)
    while True:
        tracker.step()
