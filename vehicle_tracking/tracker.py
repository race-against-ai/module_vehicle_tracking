
"""This module is used to track the car in the camera stream/video."""
# Copyright (C) 2023, NG:ITL

import sys
from json import load, dump
from time import time
from pathlib import Path
from typing import Any
from math import floor

from jsonschema import validate
import numpy as np
import cv2

from vehicle_tracking.image_sources import VideoFileSource, CameraStreamSource
from vehicle_tracking.topview_transformation import TopviewTransformation
from vehicle_tracking.publishing_handler import PublishingHandler
from tests.mocks.virtual_camera import VirtualCamera


CURRENT_DIR = Path(__file__).parent


def calculate_distance(rect1: list[int], rect2: list[int]) -> int:
    """Calculates the distance between the middle points of two rectangles.

    Args:
        rect1 (List[int]): The coordinates of the first rectangle. (x, y, w, h)
        rect2 (List[int]): The coordinates of the second rectangle. (x, y, w, h)

    Returns:
        int: The distance of the 2 rectangles.
    """
    middle1 = (rect1[0] + rect1[2] / 2, rect1[1] + rect1[3] / 2)
    middle2 = (rect2[0] + rect2[2] / 2, rect2[1] + rect2[3] / 2)
    return floor(((middle2[0] - middle1[0]) ** 2 + (middle2[1] - middle1[1]) ** 2) ** 0.5)


def sorting_function_contours(contour) -> int:
    """Sorts the contours by their area.

    Args:
        contour (cv2 contour): The contour to be sorted.

    Returns:
        int: The area of the contour.
    """
    _, _, w, h = cv2.boundingRect(contour)
    return w * h


class VehicleTracker:
    """Tracks an orange car in the camera stream/video.

    Args:
        image_source (VideoFileSource | CameraStreamSource | VirtualCamera): The source of the image.
        config_path (Path, optional): The path to the config file. Defaults to Path("./vehicle_tracking_config.json").
    """

    __lower_orange = np.array((0, 0, 100), np.uint8)
    __upper_orange = np.array((55, 115, 225), np.uint8)

    def __init__(
        self,
        image_source: VideoFileSource | CameraStreamSource | VirtualCamera,
        config_path: Path = Path("./vehicle_tracking_config.json"),
        region_of_interest_path: Path = Path("./region_of_interest.json"),
    ) -> None:
        self.__image_source = image_source
        self.__last_timestamp = time()
        self.__config: dict[str, Any] = {"config_path": config_path, "region_of_interest_path": region_of_interest_path}

        self.__transformation_points: dict[str, dict[str, tuple[int, int] | tuple[float, float]]] = {
            "top_left": {"real_world": (0, 0), "image": (0, 0)},
            "top_right": {"real_world": (0, 0), "image": (0, 0)},
            "bottom_left": {"real_world": (0, 0), "image": (0, 0)},
            "bottom_right": {"real_world": (0, 0), "image": (0, 0)},
        }
        self.__region_of_interest: np.ndarray | None = None

        self.visualized_frame: np.ndarray
        self.__processed_frame: np.ndarray
        self.__current_contours: list[Any]
        self.__previous_contours: list[Any] = []

        self.topview_transformation = TopviewTransformation()

        schemas_to_open = ["roi_config", "tracker_config", "roi_config"]
        self.__schemas: dict[str, dict] = {}
        for schema in schemas_to_open:
            with open(CURRENT_DIR / f"schemas/{schema}.json", "r", encoding="utf-8") as f:
                self.__schemas[schema] = load(f)

        with open(self.__config["config_path"], "r", encoding="utf-8") as f:
            config = load(f)
            validate(config, self.__schemas["tracker_config"])
            self.__extract_starting_config(config)
            self.__extract_transformation_points(config)

        if self.__config["region_of_interest_path"].exists():
            with open(self.__config["region_of_interest_path"], "r", encoding="utf-8") as f:
                config = load(f)
                self.__extract_region_of_interest(config)

        self.__networking_handler = PublishingHandler(self)

        if self.__config["record_video"]:
            size = image_source.frame_size[:2][::-1]
            self.__output_video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, size)  # type: ignore[call-arg]

        self.__read_new_frame()
        if "vehicle_coordinates" in self.__config and "testing" in self.__config:
            x, y, w, h = self.__config["vehicle_coordinates"]
            self.bbox = [x, y, w, h]
        else:
            self.bbox = cv2.selectROI("Car Tracking", self.__frame)  # type: ignore[arg-type]

        if "testing" in self.__config:
            self.__networking_handler.stop_awaiting_request.set()

    def __extract_starting_config(self, config_path: dict) -> None:
        """Extracts the starting config from the config file.

        Args:
            config_path (dict): The config file that was loaded.
        """
        for key in ["starting_tracker", "testing_related"]:
            if key in config_path:
                for key, data in config_path[key].items():
                    self.__config[key] = data

    def __extract_region_of_interest(self, config: dict[str, Any]) -> None:
        """Extracts the region of interest from the config file.

        Args:
            config (dict[str, Any]): The config text.
        """
        if len(config) < 3:
            print("Less than 3 elements in the region of interest, not setting it.")
            return
        validate(config, self.__schemas["roi_config"])
        self.__region_of_interest = np.array(config)

    def __extract_transformation_points(self, config: dict[str, Any]) -> None:
        """Extracts the transformation points from the config file.

        Args:
            config (dict[str, Any]): The config text.
        """
        points_list = ["top_left", "top_right", "bottom_left", "bottom_right"]
        if "coordinate_transform" in config:
            for key in points_list:
                if key in config["coordinate_transform"]:
                    point_config = config["coordinate_transform"][key]
                    self.__extract_real_world_points_from_config(point_config, key)
                    self.__extract_image_points_from_config(point_config, key)
        for point in points_list:
            image_point = self.__transformation_points[point]["image"]
            image_point = (int(image_point[0]), int(image_point[1]))
            real_world_point = self.__transformation_points[point]["real_world"]
            self.topview_transformation.set_transformation_point(point, image_point, real_world_point)

    def __extract_real_world_points_from_config(self, config: dict[str, Any], key: str):
        """Extracts the real world points from the config file.

        Args:
            config (dict[str, Any]): The config dictionary.
            key (str): The key of the point.
        """
        if "real_world" in config:
            real_world_config = config["real_world"]
            if (
                len(real_world_config) == 2
                and isinstance(real_world_config[0], (float, int))
                and isinstance(real_world_config[1], (float, int))
            ):
                points = real_world_config
                self.__transformation_points[key]["real_world"] = (points[0], points[1])

    def __extract_image_points_from_config(self, config: dict[str, Any], key: str):
        """Extracts the image points from the config file.

        Args:
            config (dict[str, Any]): The config dictionary.
            key (str): The key of the point.
        """
        if "image" in config:
            image_point_config = config["image"]
            if (
                len(image_point_config) == 2
                and isinstance(image_point_config[0], int)
                and isinstance(image_point_config[1], int)
            ):
                points = image_point_config
                self.__transformation_points[key]["image"] = (points[0], points[1])

    def get_config(self) -> dict:
        """Handles the get_config request."""
        payload = {}
        if self.__region_of_interest is not None and len(self.__region_of_interest.tolist()) >= 3:
            payload["region_of_interest"] = self.__region_of_interest.tolist()

        payload["transformation_points"] = self.__transformation_points

        return payload

    def set_config(self, config: dict) -> None:
        """Sets the config of the tracker.

        Args:
            config (dict): The config to be set.
        """
        self.__region_of_interest = None
        if "region_of_interest" in config:
            self.__region_of_interest = np.array(config["region_of_interest"])
        if "transformation_points" in config:
            for point in ["top_left", "top_right", "bottom_left", "bottom_right"]:
                conf_point = config["transformation_points"][point]
                self.__transformation_points[point]["real_world"] = (
                    float(conf_point["real_world"][0]),
                    float(conf_point["real_world"][1]),
                )
                self.__transformation_points[point]["image"] = (
                    int(conf_point["image"][0]),
                    int(conf_point["image"][1]),
                )

        if self.__region_of_interest is not None and self.__region_of_interest.size >= 3:
            with open(self.__config["region_of_interest_path"], "w", encoding="utf-8") as f:
                dump(self.__region_of_interest.tolist(), f, indent=4)

    def stop_execution(self):
        """Stops the execution of the program."""
        self.__image_source.close()
        self.__networking_handler.request_server.stop_server()
        if self.__config["record_video"]:
            self.__output_video.release()
        if self.__config["testing"]:
            return
        sys.exit(-1)

    def __read_new_frame(self) -> None:
        """Reads the next frame in the camera stream/video."""
        try:
            self.__frame = self.__image_source.read_new_frame()
            if self.__frame is None:
                raise ValueError("No frame was read.")
        except (IndexError, ValueError):
            self.stop_execution()

    def __create_timestamp(self) -> None:
        """Creates a new timestamp and outputs the FPS and delta time"""
        if "testing" not in self.__config:
            current_timestamp = time()
            delta = current_timestamp - self.__last_timestamp or 1
            fps = 1.0 / delta
            print(f"d-Time={delta}; FPS={fps}")
            self.__last_timestamp = current_timestamp

    def __process_image(self) -> None:
        """Processes the image to prepare it for tracking."""
        if self.__region_of_interest is None or len(self.__region_of_interest) < 3:
            roi = self.__frame
        else:
            mask = np.zeros_like(self.__frame)
            cv2.fillPoly(mask, [self.__region_of_interest], (255, 255, 255))  # type: ignore[arg-type]
            roi = cv2.bitwise_and(self.__frame, mask)

        in_range_image = cv2.inRange(roi, self.__lower_orange, self.__upper_orange)  # type: ignore[arg-type]

        kernel = np.ones((10, 10), np.uint8)
        closing = cv2.morphologyEx(in_range_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        self.__processed_frame = closing

    def __search_for_contours(self) -> None:
        """Searches the processed image for contours"""
        contours, _ = cv2.findContours(self.__processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=sorting_function_contours, reverse=True)  # type: ignore[assignment]

        good_contours: list[Any] = []
        for contour in contours:
            rect1 = cv2.boundingRect(contour)
            if rect1[2] * rect1[3] < 100:
                break
            to_add = True
            for good_contour in good_contours.copy():
                rect2 = cv2.boundingRect(good_contour)
                if calculate_distance(rect1, rect2) <= 75:
                    to_add = False
                    break
                good_contours.append(contour)
                to_add = False
                break
            if to_add:
                good_contours.append(contour)

        self.__current_contours = good_contours

    def __make_prediction(self) -> None:
        """Makes a prediction of a contour that is most likely the car."""
        if self.__current_contours == 0:
            return

        prediction: tuple[int, list[int]] = (-1, [0, 0, 0, 0])
        for contour in self.__current_contours:
            x, y, w, h = cv2.boundingRect(contour)
            distance = calculate_distance(self.bbox, [x, y, w, h])
            if not self.__previous_contours or len(self.__previous_contours) >= len(self.__current_contours):
                if prediction == (-1, [0, 0, 0, 0]) or distance < prediction[0]:
                    prediction = (distance, [x, y, w, h])
            else:
                if distance > 100:
                    continue
                if prediction == (-1, [0, 0, 0, 0]) or distance > prediction[0]:
                    prediction = (distance, [x, y, w, h])

        if prediction:
            self.__previous_contours = self.__current_contours
            self.bbox = prediction[1]

    def __visualize_contours(self) -> None:
        """Visualizes the contours with their bounding boxes on the processed frame."""
        self.visualized_frame = self.__frame.copy()

        for contour in self.__current_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(self.visualized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        x, y, w, h = self.bbox
        cv2.rectangle(self.visualized_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    def __write_frame_to_video(self) -> None:
        """Writes the last visualized frame to the video."""
        if self.__config["record_video"]:
            self.__output_video.write(self.visualized_frame)

    def __show_frame(self) -> None:
        """Shows the tracking view."""
        if self.__config["show_tracking_view"]:
            cv2.imshow("Car Tracking", self.visualized_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                self.stop_execution()

    def step(self) -> None:
        """Executes one full step of the tracker."""
        self.__read_new_frame()
        self.__process_image()
        self.__search_for_contours()
        self.__make_prediction()
        self.__visualize_contours()
        self.__write_frame_to_video()
        self.__show_frame()
        self.__networking_handler.send_position()
        self.__networking_handler.send_processed_image()
        self.__create_timestamp()
