# Imports
from vehicle_tracking.image_sources import VideoFileSource, CameraStreamSource
from typing import List, Any, Tuple
from json import load, dumps
from pynng import Pub0
from math import floor
from time import time
from os import path
import numpy as np
import cv2

# Constants
ADDRESS_SEND_LINK = "ipc:///tmp/RAAI/vehicle_coordinates.ipc"
LOWER_ORANGE, UPPER_ORANGE = np.array((0, 0, 100)), np.array((55, 115, 225))


# Static Calculation Functions
def calculate_distance(rect1: List[int], rect2: List[int]) -> int:
    """Calculates the distance between the middle points of two rectangles.

    Args:
        `rect1 (List[int])`: The coordinates of the first rectangle. (x, y, w, h)
        `rect2 (List[int])`: The coordinates of the second rectangle. (x, y, w, h)

    Returns:
        `int`: The distance of the 2 rectangles.
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
        image_source: VideoFileSource | CameraStreamSource,
        show_tracking_view: bool = True,
        record_video: bool = False,
    ):
        """Defines the settings and initializes everything.

        Args:
            `show_tracking_view (bool, optional)`: Decides whether it should show the tracking or not. Defaults to True.
            `record_video (bool, optional)`: Decides whether to record a video or not. Defaults to False.
            `video_path (str, optional)`: If set will use the video instead of the camera stream. Defaults to "".
        """
        self.__image_source = image_source
        self.__show_tracking_view = show_tracking_view
        self.__record_video = record_video
        self.__last_timestamp = time()
        self.__previous_contours: List[Any] = []

        self.__region_of_interest: np.ndarray | None = None
        if path.isfile("region_of_interest.json"):
            with open("region_of_interest.json", "r") as f:
                self.__region_of_interest = np.array(load(f))
        else:
            print("No region of interest found. For the best results use the script 'roi_definer.py'.")

        self.__address_sender = Pub0()
        self.__address_sender.listen(ADDRESS_SEND_LINK)

        if record_video:
            size = image_source.frame_size[:2][::-1]
            self.__output_video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, size)

        self.__read_new_frame()
        self.__bbox = cv2.selectROI("Car Tracking", self.__frame)

    # Called by self.main Functions
    def __read_new_frame(self) -> None:
        """Reads the next frame in the camera stream/video."""
        self.__frame = self.__image_source.read_new_frame()

    def __create_timestamp(self) -> None:
        """Creates a new timestamp and outputs the FPS and delta time"""
        current_timestamp = time()
        delta = current_timestamp - self.__last_timestamp or 1
        fps = 1.0 / delta
        print(f"d-Time={delta}; FPS={fps}")
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
        in_range_image = cv2.inRange(roi, LOWER_ORANGE, UPPER_ORANGE)

        # Fills all the gaps making more distinct
        kernel = np.ones((10, 10), np.uint8)
        closing = cv2.morphologyEx(in_range_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        self.__processed_frame = closing

    def __search_for_contours(self) -> None:
        """Searches the processed image for contours"""
        contours, _ = cv2.findContours(self.__processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=sorting_function_contours, reverse=True)

        good_contours: List[Any] = []
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

        self.__current_contours = good_contours

    def __make_prediction(self) -> None:
        """Makes a prediction of a contour that is most likely the car."""
        if self.__current_contours == 0:
            return
        else:
            prediction: Tuple[int, List[int]] = (-1, [0, 0, 0, 0])
            for contour in self.__current_contours:
                x, y, w, h = cv2.boundingRect(contour)
                distance = calculate_distance(self.__bbox, [x, y, w, h])
                if self.__previous_contours == [] or len(self.__previous_contours) >= len(self.__current_contours):
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
            cv2.rectangle(self.__visualized_frame, (x, y, x + w, y + h), (0, 255, 0), 2)

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
            `KeyboardInterrupt`: If 'q' is pressed uses this to stop the program.
        """
        if self.__show_tracking_view:
            cv2.imshow("Car Tracking", self.__visualized_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt("User pressed 'q' to stop the visualization.")

    def __send_bbox_coordinates(self):
        """Sends the middle coordinates of the car using pynng."""
        middle = (self.__bbox[0] + self.__bbox[2] / 2, self.__bbox[1] + self.__bbox[3] / 2)
        json_str = dumps(middle)
        self.__address_sender.send(json_str.encode("utf-8"))

    # Main Function
    def main(self):
        """Is an infinite loop that goes through the camera stream/video."""
        while True:
            self.__read_new_frame()
            self.__process_image()
            self.__search_for_contours()
            self.__make_prediction()
            self.__visualize_contours()
            self.__write_frame_to_video()
            self.__show_frame()
            self.__send_bbox_coordinates()
            self.__create_timestamp()
