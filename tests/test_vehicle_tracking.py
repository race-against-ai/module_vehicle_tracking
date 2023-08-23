# Copyright (C) 2023, NG:ITL
from mocks.virtual_camera import VirtualCamera, ToDrawObject, get_path_pixels
from vehicle_tracking.tracker import VehicleTracker
from pathlib import Path
from pynng import Sub0
from json import load, loads, dump
from typing import Any
import unittest


# Constants
CURRENT_DIR = Path(__file__).parent
CAR_COLOR = (1, 70, 206)


# Reminder: Naming convention for unit tests
#
# test_InitialState_PerformedAction_ExpectedResult


class VehicleTrackingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.__config_path = CURRENT_DIR.parent / "vehicle_tracking_config.json"
        self.__config: dict[str, Any]
        with open(self.__config_path, "r") as config_file:
            self.__config = load(config_file)
        with open(self.__config_path, "w") as config_file:
            excluded: dict[str, Any] = self.__config.copy()
            excluded["roi_points"] = []
            dump(excluded, config_file, indent=4)

    def tearDown(self) -> None:
        with open(self.__config_path, "w") as config_file:
            dump(self.__config, config_file, indent=4)

    def test_CarOnly_CheckingTrackedPosition_CarTrackedSuccessfully(self) -> None:
        try:
            tracking_path: list[tuple[int, int]] = [
                (125, 127),
                (148, 267),
                (233, 311),
                (338, 341),
                (420, 201),
                (384, 61),
                (234, 57),
                (185, 83),
            ]
            actual_path: list[tuple[int, int]] = get_path_pixels(tracking_path)
            car_draw_object = ToDrawObject(CAR_COLOR, [(0, 0), (40, 0), (40, 40), (0, 40)], 1, actual_path)
            wall = ToDrawObject((255, 255, 255), [(0, 0), (20, 0), (20, 20), (0, 20)], 0, [(0, 0)])
            start_coord = actual_path[0]
            middle = car_draw_object.centroid
            initial_position: tuple[int, int, int, int] = (
                start_coord[0] - middle[0],
                start_coord[1] - middle[1],
                start_coord[0] + middle[0],
                start_coord[1] + middle[1],
            )
            source = VirtualCamera([car_draw_object, wall], 60)
            tracker = VehicleTracker(source, vehicle_coordinates=initial_position)
            self.__coordinate_sub = Sub0()
            sub_address = self.__config["pynng"]["publishers"]["position_sender"]["address"]
            sub_topic = self.__config["pynng"]["publishers"]["position_sender"]["topics"]["coords_image"]
            self.__coordinate_sub.subscribe(sub_topic)
            self.__coordinate_sub.dial(sub_address)
            passed = True
            for i in range(len(actual_path)):
                tracker.step()
                coord_bytes: bytes = self.__coordinate_sub.recv()
                coord_str: str = coord_bytes.decode("utf-8")
                coord_str = coord_str.split(" ", maxsplit=1)[1]
                coord = loads(coord_str)
                x_offset, y_offset = coord[0] - actual_path[i][0], coord[1] - actual_path[i][1]
                print(coord)
                if x_offset > 5 or y_offset > 5:
                    passed = False
                    break
        except Exception as e:
            print(e)
            self.assertTrue(False)
            return

        self.assertTrue(passed)
