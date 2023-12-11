"""This module is used to track the car in the camera stream/video."""
# Copyright (C) 2023, NG:ITL

import sys
from threading import Thread, Event
from json import load, loads, dumps
from pathlib import Path
from typing import Any
from math import floor
from time import time

from pynng import Pub0, Rep0, Timeout
import numpy as np
import cv2

from vehicle_tracking.image_sources import VideoFileSource, CameraStreamSource
from vehicle_tracking.topview_transformation import TopviewTransformation
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
    """Sorts the contours by the size of the bounding box."""
    _, _, w, h = cv2.boundingRect(contour)
    return w * h


class VehicleTracker:
    """This class is used to track the car in the camera stream/video.

    Args:
        show_tracking_view (bool, optional): Decides whether it should show the tracking or not. Defaults to True.
        record_video (bool, optional): Decides whether to record a video or not. Defaults to False.
        video_path (str, optional): If set will use the video instead of the camera stream. Defaults to "".
        vehicle_coordinates (None | Tuple[int, int, int, int], optional): If set it will not prompt the selection of the car. Defaults to None. (Testing)
        config_path (Path, optional): The path to the config file. Defaults to CURRENT_DIR.parent / "vehicle_tracking_config.json".
        testing (bool, optional): If set to True will not print timestamps. Defaults to False.
    """

    __lower_orange = np.array((0, 0, 100), np.uint8)
    __upper_orange = np.array((55, 115, 225), np.uint8)

    def __init__(
        self,
        image_source: VideoFileSource | CameraStreamSource | VirtualCamera,
        show_tracking_view: bool = True,
        record_video: bool = False,
        vehicle_coordinates: None | tuple[int, int, int, int] = None,
        config_path: Path = CURRENT_DIR.parent / "vehicle_tracking_config.json",
        testing: bool = False,
    ) -> None:
        self.__image_source = image_source
        self.__show_tracking_view = show_tracking_view
        self.__record_video = record_video
        self.__testing = testing
        self.__last_timestamp = time()

        self.__visualized_frame: np.ndarray
        self.__processed_frame: np.ndarray
        self.__current_contours: list[Any]
        self.__previous_contours: list[Any] = []

        self.__topview_transformation = TopviewTransformation()

        self.__transformation_points: dict[str, dict[str, tuple[int, int] | tuple[float, float]]] = {
            "top_left": {"real_world": (0, 0), "image": (0, 0)},
            "top_right": {"real_world": (0, 0), "image": (0, 0)},
            "bottom_left": {"real_world": (0, 0), "image": (0, 0)},
            "bottom_right": {"real_world": (0, 0), "image": (0, 0)},
        }
        self.__region_of_interest: np.ndarray | None = None

        self.__position_sender_link: str
        self.__position_sender_topics: dict[str, str]
        self.__processed_frame_link: str
        self.__config_handler_link: str

        with open(config_path, "r", encoding="utf-8") as f:
            config = load(f)
            self.__extract_pynng_config(config)
            self.__extract_region_of_interest(config)
            self.__extract_transformation_points(config)

        self.__position_sender = Pub0(listen=self.__position_sender_link)
        self.__frame_sender = Pub0(listen=self.__processed_frame_link)
        self.__config_handler = Rep0(listen=self.__config_handler_link)

        if record_video:
            size = image_source.frame_size[:2][::-1]
            self.__output_video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, size)

        self.__read_new_frame()
        if vehicle_coordinates:
            x, y, w, h = vehicle_coordinates
            self.__bbox = [x, y, w, h]
        else:
            self.__bbox = cv2.selectROI("Car Tracking", self.__frame)

        if not self.__testing:
            self.__stop_awaiting_request = Event()
            self.__wait_for_request_thread = Thread(target=self.__wait_for_request)
            self.__wait_for_request_thread.start()

    def __extract_pynng_config(self, config: dict[str, Any]) -> None:
        """Extracts the pynng config from the config file.

        Args:
            config (dict[str, Any]): The config file.
        """
        pubs = config["pynng"]["publishers"]
        self.__position_sender_link = pubs["position_sender"]["address"]
        self.__position_sender_topics = pubs["position_sender"]["topics"]
        self.__processed_frame_link = pubs["processed_image_sender"]["address"]
        self.__config_handler_link = pubs["request_config_sender"]["address"]

    def __extract_region_of_interest(self, config: dict[str, Any]) -> None:
        if "roi_points" in config:
            if len(config["roi_points"]) >= 3:
                if all(isinstance(x, float) and isinstance(y, float) for x, y in config["roi_points"]):
                    points = [(int(x), int(y)) for x, y in config["roi_points"]]
                    self.__region_of_interest = np.array(points)
                else:
                    raise TypeError("The roi_points in the config file are not in the correct format.")
            else:
                return

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
            self.__topview_transformation.set_transformation_point(point, image_point, real_world_point)

    def __extract_real_world_points_from_config(self, config, key):
        real_world_config = config["real_world_points"]
        if (
            "real_world_points" in config
            and len(real_world_config) == 2
            and isinstance(real_world_config[0], (float, int))
            and isinstance(real_world_config[1], (float, int))
        ):
            points = real_world_config
            self.__transformation_points[key]["real_world"] = (points[0], points[1])
        elif len(real_world_config) != 0:
            raise TypeError(f"The real_world_points in the {key} transformation are not in the correct format.")

        return config

    def __extract_image_points_from_config(self, config, key):
        image_point_config = config["image_points"]
        if (
            "image_points" in config
            and len(image_point_config) == 2
            and isinstance(image_point_config[0], int)
            and isinstance(image_point_config[1], int)
        ):
            points = image_point_config
            self.__transformation_points[key]["image"] = (points[0], points[1])
        elif len(image_point_config) != 0:
            raise TypeError(f"The image_points in the {key} transformation are not in the correct format.")

    def __wait_for_request(self) -> None:
        """Waits for a request from the configuration interface."""
        while not self.__stop_awaiting_request.is_set():
            request = self.__config_handler.recv()
            command, data = request.split(b" ", 1)
            if command == b"REQUEST":
                conf = {
                    "Region of Interest": self.__region_of_interest.tolist()
                    if self.__region_of_interest is not None
                    else None,
                    "Transformation Points": self.__transformation_points,
                }
                self.__config_handler.send(b"RETURN_DATA " + dumps(conf).encode("utf-8"))
            elif command == b"UPDATE":
                conf = loads(data.decode("utf-8"))
                self.__region_of_interest = np.array(conf["Region of Interest"])
                for point in ["top_left", "top_right", "bottom_left", "bottom_right"]:
                    conf_point = conf["Transformation Points"][point]
                    self.__transformation_points[point]["real_world"] = (
                        float(conf_point["real_world"][0]),
                        float(conf_point["real_world"][1]),
                    )
                    self.__transformation_points[point]["image"] = (
                        int(conf_point["image"][0]),
                        int(conf_point["image"][1]),
                    )
                self.__config_handler.send(b"OK NONE")
            elif command == b"OK":
                print("Received OK from configurator")
            elif command == b"PING":
                print("Pong")
                self.__config_handler.send(b"PONG")
            else:
                self.__config_handler.send(b"ERROR Unknown command")

    def stop_execution(self):
        self.__stop_awaiting_request.set()
        self.__wait_for_request_thread.join()
        sys.exit(1)

    def __read_new_frame(self) -> None:
        """Reads the next frame in the camera stream/video."""
        try:
            self.__frame = self.__image_source.read_new_frame()
            if self.__frame is None:
                self.stop_execution()
        except IndexError:
            self.stop_execution()

    def __create_timestamp(self) -> None:
        """Creates a new timestamp and outputs the FPS and delta time"""
        if not self.__testing:
            current_timestamp = time()
            delta = current_timestamp - self.__last_timestamp or 1
            fps = 1.0 / delta
            print(f"d-Time={delta}; FPS={fps}")
            self.__last_timestamp = current_timestamp

    def __process_image(self) -> None:
        """Processes the image to prepare it for tracking."""
        # Region of Interest
        if self.__region_of_interest is None or len(self.__region_of_interest) < 3:
            roi = self.__frame
        else:
            mask = np.zeros_like(self.__frame)
            cv2.fillPoly(mask, [self.__region_of_interest], (255, 255, 255))
            roi = cv2.bitwise_and(self.__frame, mask)

        # Filters color to be mostly orange
        in_range_image = cv2.inRange(roi, self.__lower_orange, self.__upper_orange)

        # Fills all the gaps making more distinct
        kernel = np.ones((10, 10), np.uint8)
        closing = cv2.morphologyEx(in_range_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        self.__processed_frame = closing

    def __search_for_contours(self) -> None:
        """Searches the processed image for contours"""
        contours, _ = cv2.findContours(self.__processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=sorting_function_contours, reverse=True)

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
        else:
            prediction: tuple[int, list[int]] = (-1, [0, 0, 0, 0])
            for contour in self.__current_contours:
                x, y, w, h = cv2.boundingRect(contour)
                distance = calculate_distance(self.__bbox, [x, y, w, h])
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
                self.__bbox = prediction[1]

    def __visualize_contours(self) -> None:
        """Visualizes the contours with their bounding boxes on the processed frame."""
        self.__visualized_frame = self.__frame.copy()

        # Draws green boxes around each possibility.
        for contour in self.__current_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(self.__visualized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draws a white box around the car
        x, y, w, h = self.__bbox
        cv2.rectangle(self.__visualized_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    def __write_frame_to_video(self) -> None:
        """Writes the last visualized frame to the video."""
        if self.__record_video:
            self.__output_video.write(self.__visualized_frame)

    def __show_frame(self) -> None:
        """Shows the last visualized frame and allows to exit the application with 'q'.

        Raises:
            KeyboardInterrupt: If 'q' is pressed uses this to stop the program.
        """
        if self.__show_tracking_view:
            cv2.imshow("Car Tracking", self.__visualized_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                self.stop_execution()

    def __send_bbox_coordinates(self) -> None:
        """Sends the middle coordinates of the car using pynng."""
        middle = (
            self.__bbox[0] + floor(self.__bbox[2] / 2),
            self.__bbox[1] + floor(self.__bbox[3] / 2),
        )
        str_with_topic = self.__position_sender_topics["coords_image"] + " " + dumps(middle)
        self.__position_sender.send(str_with_topic.encode("utf-8"))

    def __send_world_coordinates(self) -> None:
        """Sends the world coordinates of the car using pynng."""
        middle = (
            self.__bbox[0] + floor(self.__bbox[2] / 2),
            self.__bbox[1] + floor(self.__bbox[3] / 2),
        )
        real_world_coords = self.__topview_transformation.image_to_world_transform(middle)
        str_with_topic = self.__position_sender_topics["coords_world"] + " " + dumps(real_world_coords)
        self.__position_sender.send(str_with_topic.encode("utf-8"))

    def __send_processed_frame(self) -> None:
        """Sends the processed frame to time_tracking using pynng."""
        np_frame = np.array(self.__visualized_frame)
        frame_bytes = np_frame.tobytes()
        self.__frame_sender.send(frame_bytes)

    def step(self) -> None:
        """Is an infinite loop that goes through the camera stream/video."""
        self.__read_new_frame()
        # print(1)
        self.__process_image()
        # print(2)
        self.__search_for_contours()
        # print(3)
        self.__make_prediction()
        # print(4)
        self.__visualize_contours()
        # print(5)
        self.__write_frame_to_video()
        # print(6)
        self.__show_frame()
        # print(7)
        self.__send_bbox_coordinates()
        # print(8)
        self.__send_world_coordinates()
        # print(9)
        self.__send_processed_frame()
        # print(10)
        self.__create_timestamp()
        # print(11)
