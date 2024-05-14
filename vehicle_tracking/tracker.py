# Copyright (C) 2023, NG:ITL
from vehicle_tracking.image_sources import VideoFileSource, CameraStreamSource
from tests.mocks.virtual_camera import VirtualCamera
from typing import Any
from json import load, dumps
from pathlib import Path
from pynng import Pub0
from math import floor
from time import time
import numpy as np
import cv2

# Constants
CURRENT_DIR = Path(__file__).parent


# Static Calculation Functions
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


# Static Sorting Functions
def sorting_function_contours(contour) -> int:
    """Sorts the contours by the size of the bounding box."""
    _, _, w, h = cv2.boundingRect(contour)
    return w * h


# Classes
class VehicleTracker:
    # Initialization
    def __init__(
        self,
        image_source: VideoFileSource | CameraStreamSource | VirtualCamera,
        show_tracking_view: bool = True,
        record_video: bool = False,
        vehicle_coordinates: None | tuple[int, int, int, int] = None,
        config_path: Path = CURRENT_DIR.parent / "vehicle_tracking_config.json",
    ):
        """Defines the settings and initializes everything.

        Args:
            show_tracking_view (bool, optional): Decides whether it should show the tracking or not. Defaults to True.
            record_video (bool, optional): Decides whether to record a video or not. Defaults to False.
            video_path (str, optional): If set will use the video instead of the camera stream. Defaults to "".
            vehicle_coordinates (None | Tuple[int, int, int, int], optional): If set it will not prompt the selection of the car. Defaults to None. (Testing)
            config_path (Path, optional): The path to the config file. Defaults to CURRENT_DIR.parent / "vehicle_tracking_config.json".
        """
        self.__image_source = image_source
        self.__show_tracking_view = show_tracking_view
        self.__record_video = record_video
        self.__last_timestamp = time()
        self.__previous_contours: list[list[Any]] = [[],[]]
        self.__lower_orange = np.array((0, 0, 100), np.uint8)
        self.__upper_orange = np.array((55, 115, 225), np.uint8)
        self.__lower_green = np.array((0, 150, 0), np.uint8)
        self.__upper_green = np.array((100, 255, 100), np.uint8)
        self.car_to_show = 0

        self.__region_of_interest: np.ndarray | None = None
        with open(config_path, "r") as f:
            config = load(f)
            self.__position_sender_link: str = config["pynng"]["publishers"]["position_sender"]["address"]
            self.__position_sender_topics: dict[str, str] = config["pynng"]["publishers"]["position_sender"]["topics"]
            self.__processed_frame_link: str = config["pynng"]["publishers"]["processed_image_sender"]["address"]
            if "roi_points" in config and len(config["roi_points"]) >= 3:
                self.__region_of_interest = np.array(config["roi_points"])
            else:
                print(
                    f"No region of interest found. Please use the '{CURRENT_DIR / 'roi_definer.py'}' to select the region of interest and restart the tracker."
                )

        self.__position_sender = Pub0(listen=self.__position_sender_link)
        self.__frame_sender = Pub0(listen=self.__processed_frame_link)

        if record_video:
            size = image_source.frame_size[:2][::-1]
            self.__output_video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, size)

        self.__read_new_frame()
        self.__bbox = []
        self.__bbox.append(cv2.selectROI("Car Tracking", self.__frame))
        self.__bbox.append(cv2.selectROI("Car Tracking", self.__frame))
        # if vehicle_coordinates:
        #     x, y, w, h = vehicle_coordinates
        #     self.__bbox = [x, y, w, h]
        # else:
        #     self.__bbox = []
        #     print(cv2.selectROI("Car Tracking", self.__frame))
        #     self.__bbox[0] = cv2.selectROI("Car Tracking", self.__frame)
        #     self.__bbox[1] = cv2.selectROI("Car Tracking", self.__frame)

    # Called by self.main Functions
    def __read_new_frame(self) -> None:
        """Reads the next frame in the camera stream/video."""
        self.__frame = self.__image_source.read_new_frame()

    def __create_timestamp(self) -> None:
        """Creates a new timestamp and outputs the FPS and delta time"""
        current_timestamp = time()
        delta = current_timestamp - self.__last_timestamp or 1
        fps = 1.0 / delta
        #print(f"d-Time={delta}; FPS={fps}")
        self.__last_timestamp = current_timestamp

    def __process_image(self) -> None:
        """Processes the image to prepare it for tracking."""
        # Region of Interest
        if self.__region_of_interest is None:
            roi = self.__frame
        else:
            mask = np.zeros_like(self.__frame)
            cv2.fillPoly(mask, [self.__region_of_interest], (255, 255, 255))
            roi = cv2.bitwise_and(self.__frame, mask)

        # Filters color to be mostly orange
        in_range_image_orange = cv2.inRange(roi, self.__lower_orange, self.__upper_orange)
        in_range_image_green = cv2.inRange(roi, self.__lower_green, self.__upper_green)

        # Fills all the gaps making more distinct
        kernel = np.ones((10, 10), np.uint8)
        closing_orange = cv2.morphologyEx(in_range_image_orange.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        closing_green = cv2.morphologyEx(in_range_image_green.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        

        self.__processed_frames = [closing_orange, closing_green]

    def __search_for_contours(self) -> None:
        """Searches the processed image for contours"""
        self.__current_contours = []
        for processed_frame in self.__processed_frames:
            contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=sorting_function_contours, reverse=True)

            good_contours: list[Any] = []
            for contour in contours:
                rect1 = cv2.boundingRect(contour)
                if rect1[2] * rect1[3] < 100:
                    break
                to_add = True
                for good_contour in good_contours:
                    rect2 = cv2.boundingRect(good_contour)
                    if calculate_distance(rect1, rect2) <= 75:
                        to_add = False
                        break
                    good_contours.append(contour)
                    to_add = False
                    break
                if to_add:
                    good_contours.append(contour)

            #first contour is orange car, second green
                
            self.__current_contours.append(good_contours)

    def __make_prediction(self) -> None:
        """Makes a prediction of a contour that is most likely the car."""
        #first contour is orange car, second green
        for index, current_contour in enumerate(self.__current_contours):
            if current_contour == 0:
                return
            else:
                prediction: tuple[int, list[int]] = (-1, [0, 0, 0, 0])
                for contour in current_contour:
                    x, y, w, h = cv2.boundingRect(contour)
                    distance = calculate_distance(self.__bbox[index], [x, y, w, h])
                    if self.__previous_contours == [] or len(self.__previous_contours) >= len(current_contour):
                        if prediction == (-1, [0, 0, 0, 0]) or distance < prediction[0]:
                            prediction = (distance, [x, y, w, h])
                    else:
                        if distance > 100:
                            continue
                        if prediction == (-1, [0, 0, 0, 0]) or distance > prediction[0]:
                            prediction = (distance, [x, y, w, h])

                if prediction:
                    self.__previous_contours[index] = current_contour
                    self.__bbox[index] = prediction[1]
                    print(self.__bbox)
                    print(prediction)

    def __visualize_contours(self) -> None:
        """Visualizes the contours with their bounding boxes on the processed frame."""
        self.__visualized_frame = self.__frame.copy()
        
        for index, current_contour in enumerate(self.__current_contours):
            # Draws green boxes around each possibility.
            for contour in current_contour:
                x, y, w, h = cv2.boundingRect(contour)
                if index == 0: #orange car
                    cv2.rectangle(self.__visualized_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else: 
                    cv2.rectangle(self.__visualized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if index == self.car_to_show:
                x, y, w, h = self.__bbox[index]
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
                raise KeyboardInterrupt("User pressed 'q' to stop the visualization.")

    def __send_bbox_coordinates(self):
        print(self.__bbox)
        """Sends the middle coordinates of the car using pynng."""
        middle = (self.__bbox[self.car_to_show][0] + floor(self.__bbox[self.car_to_show][2] / 2), self.__bbox[self.car_to_show][1] + floor(self.__bbox[self.car_to_show][3] / 2))
        print(middle)
        str_with_topic = self.__position_sender_topics["coords_image"] + " " + dumps(middle)
        self.__position_sender.send(str_with_topic.encode("utf-8"))

    def __send_processed_frame(self):
        """Sends the processed frame to time_tracking using pynng."""
        np_frame = np.array(self.__visualized_frame)
        frame_bytes = np_frame.tobytes()
        self.__frame_sender.send(frame_bytes)


    # Main Function
    def step(self):
        """Is an infinite loop that goes through the camera stream/video."""

        # Create a black image
        img = 255 * np.ones((150, 400, 3), dtype=np.uint8)

        # Draw two buttons
        cv2.rectangle(img, (50, 50), (150, 100), (0, 255, 0), -1)
        cv2.rectangle(img, (200, 50), (300, 100), (0, 0, 255), -1)

        # Put text on buttons
        cv2.putText(img, 'Green', (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img, 'Orange', (210, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Create a window and set the mouse callback function
        cv2.namedWindow('car to time track')
        cv2.setMouseCallback('car to time track', self.mouse_callback)

        while True:

            cv2.imshow('car to time track', img)

            self.__read_new_frame()
            self.__process_image()
            self.__search_for_contours()
            self.__make_prediction()
            self.__visualize_contours()
            self.__write_frame_to_video()
            self.__show_frame()
            self.__send_bbox_coordinates()
            self.__send_processed_frame()
            self.__create_timestamp()


    # Function to handle mouse clicks
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 50 <= x <= 150 and 50 <= y <= 100:
                self.car_to_show = 1
            elif 200 <= x <= 300 and 50 <= y <= 100:
                self.car_to_show = 0

