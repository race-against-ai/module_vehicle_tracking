"""Unit tests for the VehicleTracking class."""
# Copyright (C) 2023, NG:ITL

from json import load, loads, dump
import unittest
from pathlib import Path

from pynng import Sub0

from tests.mocks.virtual_camera import VirtualCamera, ToDrawObject, get_path_pixels
from vehicle_tracking.tracker import VehicleTracker


CURRENT_DIR = Path(__file__).parent
CAR_COLOR = (1, 70, 206)
TEMPLATE_CONFIG_PATH = CURRENT_DIR.parent / "vehicle_tracking/templates/tracker_config.json"
TESTING_CONFIG_PATH = CURRENT_DIR / "tmp/vehicle_tracking_config.json"
TESTING_REGION_OF_INTEREST_PATH = CURRENT_DIR / "tmp/region_of_interest.json"


class VehicleTrackingTest(unittest.TestCase):
    """Tests for the VehicleTracking."""

    def setUp(self) -> None:
        if not TESTING_CONFIG_PATH.parent.exists():
            TESTING_CONFIG_PATH.parent.mkdir()

        with open(TEMPLATE_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            self.__config = load(config_file)
        with open(TESTING_CONFIG_PATH, "w", encoding="utf-8") as config_file:
            conf = self.__config.copy()
            conf["transformation_points"] = (
                {
                    "top_left": {"real_world": [], "image": []},
                    "top_right": {"real_world": [], "image": []},
                    "bottom_left": {"real_world": [], "image": []},
                    "bottom_right": {"real_world": [], "image": []},
                },
            )
            dump(conf, config_file, indent=4)
            print(conf["pynng"]["publishers"])
        with open(TESTING_REGION_OF_INTEREST_PATH, "w", encoding="utf-8") as roi_file:
            dump([], roi_file, indent=4)

    def tearDown(self) -> None:
        # pass
        if TESTING_CONFIG_PATH.exists():
            TESTING_CONFIG_PATH.unlink()
        if TESTING_CONFIG_PATH.parent.exists():
            TESTING_REGION_OF_INTEREST_PATH.unlink()
        if TESTING_CONFIG_PATH.parent.exists():
            TESTING_CONFIG_PATH.parent.rmdir()

    def test_car_only_checking_tracked_position_car_tracked_successfully(self) -> None:
        """Tests that the car is tracked successfully. It will draw an orange rectangle on the screen representing the car."""
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

        with open(TESTING_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            config = load(config_file)

        config["testing_related"] = {"testing": True, "vehicle_coordinates": list(initial_position)}
        config["starting_tracker"]["show_tracking_view"] = False

        with open(TESTING_CONFIG_PATH, "w", encoding="utf-8") as config_file:
            dump(config, config_file, indent=4)

        source = VirtualCamera([car_draw_object], 60)
        tracker = VehicleTracker(
            source, config_path=TESTING_CONFIG_PATH, region_of_interest_path=TESTING_REGION_OF_INTEREST_PATH
        )
        position_sender = self.__config["pynng"]["publishers"]["position_sender"]
        sub_address = position_sender["address"]
        sub_topic = position_sender["topics"]["coords_image"]
        coordinate_sub = Sub0(topics=sub_topic, dial=sub_address)

        passed = True
        for point in actual_path:
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
        tracker.stop_execution()
