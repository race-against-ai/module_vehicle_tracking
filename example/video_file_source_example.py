# Copyright (C) 2023 NG:ITL
from vehicle_tracking.image_sources import VideoFileSource
from vehicle_tracking.tracker import VehicleTracker
from pathlib import Path
import sys


# Constants
FILE_DIR = Path(__file__).parent
VIDEO_FILE_SOURCE_FILEPATH = FILE_DIR.parent / "resources/test_video_1.h265"


if __name__ == "__main__":
    if not VIDEO_FILE_SOURCE_FILEPATH.exists():
        print("Please run download script in 'resources/download_resources.py'")
        sys.exit(-1)

    source = VideoFileSource(VIDEO_FILE_SOURCE_FILEPATH, frame_rate=60)
    tracker = VehicleTracker(source)
    while True:
        tracker.step()
