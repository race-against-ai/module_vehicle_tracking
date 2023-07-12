# Copyright (C) 2023, NG:ITL
import unittest
from pathlib import Path

from vehicle_tracking.image_sources import CameraStreamSource, VideoFileSource
from vehicle_tracking.tracker import VehicleTracker

FILE_DIR = Path(__file__).parent
VIDEO_FILE_SOURCE_FILEPATH = FILE_DIR.parent / "resources/example_video.avi"

# Reminder: Naming convention vor unit tests
#
# test_InitialState_PerformedAction_ExpectedResult

class VehicleTrackingTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_InitialState_PerformedAction_ExpectedResult(self) -> None:
        source = VideoFileSource(VIDEO_FILE_SOURCE_FILEPATH, frame_rate=30)
        tracker = VehicleTracker(source)

        tracker.step()



    def test_InitialState2_PerformedAction_ExpectedResult(self) -> None:
        pass
