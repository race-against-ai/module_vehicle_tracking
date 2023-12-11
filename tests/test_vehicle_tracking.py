"""Unit tests for the VehicleTracking class."""
# Copyright (C) 2023, NG:ITL

from os import mkdir, remove, rmdir
from json import load, loads, dump
import unittest
from pathlib import Path

from pynng import Sub0

from mocks.virtual_camera import VirtualCamera, ToDrawObject, get_path_pixels
from vehicle_tracking.tracker import VehicleTracker


# Constants
CURRENT_DIR = Path(__file__).parent
CAR_COLOR = (1, 70, 206)


# Reminder: Naming convention for unit tests
#
# test_InitialState_PerformedAction_ExpectedResult


class VehicleTrackingTest(unittest.TestCase):
    """Tests for the VehicleTracking."""

    def setUp(self) -> None:
        self.__config_path = CURRENT_DIR.parent / "vehicle_tracking_config.json"
        self.__testing_config_path = CURRENT_DIR / "tmp/vehicle_tracking_config.json"
        if not self.__testing_config_path.parent.exists():
            mkdir(self.__testing_config_path.parent)

        with open(self.__config_path, "r", encoding="utf-8") as config_file:
            self.__config = load(config_file)
        with open(self.__testing_config_path, "w", encoding="utf-8") as config_file:
            conf = self.__config.copy()
            conf["roi_points"] = []
            dump(conf, config_file, indent=4)

    def tearDown(self) -> None:
        remove(self.__testing_config_path)
        rmdir(self.__testing_config_path.parent)

    def test_car_only_checking_tracked_position_car_tracked_successfully(self) -> None:
        """Tests that the car is tracked successfully. It will draw an orange rectangle on the representing the car."""
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
        start_coord = actual_path[0]
        middle = car_draw_object.centroid
        initial_position: tuple[int, int, int, int] = (
            start_coord[0] - middle[0],
            start_coord[1] - middle[1],
            start_coord[0] + middle[0],
            start_coord[1] + middle[1],
        )

        source = VirtualCamera([car_draw_object], 60)
        tracker = VehicleTracker(
            source,
            show_tracking_view=False,
            vehicle_coordinates=initial_position,
            config_path=self.__testing_config_path,
            testing=True,
        )
        position_sender = self.__config["pynng"]["publishers"]["position_sender"]
        sub_address = position_sender["address"]
        sub_topic = position_sender["topics"]["coords_image"]
        coordinate_sub = Sub0(topics=sub_topic, dial=sub_address)

        passed = True
        # x = 0
        # print(len(actual_path))
        for point in actual_path:
            # print(x)
            # x += 1
            tracker.step()

            coord_bytes: bytes = coordinate_sub.recv()
            coord_str: str = coord_bytes.decode("utf-8")
            coord_str = coord_str.split(" ", maxsplit=1)[1]
            coord = loads(coord_str)

            x_offset = abs(coord[0] - point[0])
            y_offset = abs(coord[1] - point[1])
            if x_offset > 5 or y_offset > 5:
                passed = False
                break
        self.assertTrue(passed)
