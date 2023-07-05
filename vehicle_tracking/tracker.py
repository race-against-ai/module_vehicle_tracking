# Imports
from json import load, dumps
from typing import List
from pynng import Pub0, Sub0
from time import time
from os import path
import numpy as np
import cv2

# Constants
ADDRESS_SEND_LINK = "ipc:///tmp/RAAI/vehicle_coordinates.ipc"
FRAME_RECEIVE_LINK = "ipc:///tmp/RAAI/camera_frame.ipc"
CURRENT_DIR = path.dirname(path.abspath(__file__))


# Static Calculation Functions
def calculate_distance(rect1: List[int], rect2: List[int]) -> float:
    """
    Calculates the distance between the 2 middle points of the rectangles.

    Input:
    `rect1: List[int]` -> Points of first rectangle.
    `rect2: List[int]` -> Points of second rectangle.

    Output:
    distance:float -> distance between points.
    """
    middle1 = (rect1[0] + rect1[2] / 2,
               rect1[1] + rect1[3] / 2)
    middle2 = (rect2[0] + rect2[2] / 2,
               rect2[1] + rect2[3] / 2)
    return ((middle2[0] - middle1[0]) ** 2 + (middle2[1] - middle1[1]) ** 2) ** 0.5


# Static Sorting Functions
def sorting_function_contours(contour):
    """
    Sorts the contours by the size of the bounding box.
    """
    _, _, w, h = cv2.boundingRect(contour)
    return w * h


# Classes
class VehicleTracker:
    # Initialization
    def __init__(self, show_tracking_view: bool = True, record_video: bool = False, video_path: str = ""):
        """
        Initializes the class.
        
        Input:
        `show_tracking_view: bool = True` -> Sets weather the visualized image should be shown.
        `record_video: bool = False` -> Defines a video should be recorded or not.
        
            Debugging Only (!!DO NOT USE!!):
            `video_path: str = ""` -> If use_camera_stream is set to false it will try to read this video.
        
        Output:
        `None`
        """
        self.__SHOW_TRACKING_VIEW = show_tracking_view
        self.__RECORD_VIDEO = record_video
        self.__VIDEO_PATH = video_path
        self.__LOWER_ORANGE, self.__UPPER_ORANGE = np.array((0, 0, 100)), np.array((55, 115, 225))
        self.__define_cap()
        self.__define_roi()
        self.__define_coordinate_sender()
        if record_video:
            self.__OUTPUT_VIDEO = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, self.__VIDEO_SIZE)
        self.__bbox = cv2.selectROI("Car Tracking", self.__frame)
        self.__last_timestamp = time()
        self.__previous_contours = []

    def __define_cap(self) -> None:
        """
        Defines the video cap.
        
        Input/Output:
        `None`
        """
        if self.__VIDEO_PATH == "":
            self.__define_image_receiver()
            self.__VIDEO_SIZE = (640, 480)
            self.__RESHAPE_VIDEO_SIZE = (480, 640, 3)
            self.__read_new_frame()
        else:
            self.__VIDEO_CAP = cv2.VideoCapture(self.__VIDEO_PATH)
            success, self.__frame = self.__VIDEO_CAP.read()
            self.__VIDEO_SIZE = self.__frame.shape
            if not success:
                raise FileNotFoundError(
                    "Could not read the first frame of the video. Please try again and validate the video path.")

            self.__VIDEO_SIZE = self.__frame.shape[:2][::-1]

    def __define_roi(self) -> None:
        """
        Defines the roi list.
        
        Input/Output:
        `None`
        """
        if path.isfile(path.join(CURRENT_DIR, "region_of_interest.json")):
            with open(path.join(CURRENT_DIR, "region_of_interest.json"), "r") as f:
                self.__REGION_OF_INTEREST = np.array(load(f))
        else:
            self.__REGION_OF_INTEREST = None
            print("No region of interest found. For the best results use the script 'roi_definer.py'.")

    def __define_coordinate_sender(self):
        """
        Defines the pynng sender for the coordinates.
        
        Input/Output:
        `None`
        """
        self.__ADDRESS_SENDER = Pub0()
        self.__ADDRESS_SENDER.listen(ADDRESS_SEND_LINK)

    def __define_image_receiver(self) -> None:
        """
        Defines the image receiver, so it can receive the camera's images.
        
        Input/Output:
        `None`
        """
        self.__FRAME_RECEIVER = Sub0()
        self.__FRAME_RECEIVER.subscribe("")
        self.__FRAME_RECEIVER.dial(FRAME_RECEIVE_LINK)

    # Called by self.main Functions
    def __read_new_frame(self) -> None:
        """
        Reads the next frame in the video.
        
        Input/Output:
        `None`
        """
        if self.__VIDEO_PATH == "":
            frame_bytes = self.__FRAME_RECEIVER.recv()
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            self.__frame = frame.reshape(self.__RESHAPE_VIDEO_SIZE)
        else:
            success, self.__frame = self.__VIDEO_CAP.read()
            if not success:
                raise IndexError("Frame could not be read. Video probably ended.")

    def __create_timestamp(self) -> None:
        """
        Creates a new timestamp and if show_output will print FPS and delta time.
        
        Input/Output:
        `None` 
        """
        current_timestamp = time()
        delta = current_timestamp - self.__last_timestamp or 1
        fps = 1.0 / delta
        print(f"d-Time={delta}; FPS={fps}")
        self.__last_timestamp = current_timestamp

    def __process_image(self) -> None:
        """
        Process the image to prepare it for contour tracking.
        
        Input/Output:
        `None`
        """
        # Region of Interest
        if self.__REGION_OF_INTEREST is None:
            roi = self.__frame
        else:
            mask = np.zeros_like(self.__frame)
            cv2.fillPoly(mask, [self.__REGION_OF_INTEREST], (255, 255, 255))
            roi = cv2.bitwise_and(self.__frame, mask)

        # Filters color to be mostly orange
        in_range_image = cv2.inRange(roi, self.__LOWER_ORANGE, self.__UPPER_ORANGE)

        # Fills all the gaps making more distinct 
        kernel = np.ones((10, 10), np.uint8)
        closing = cv2.morphologyEx(in_range_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        self.__processed_frame = closing

    def __search_for_contours(self) -> None:
        """
        Searches the processed image for contours.
        
        Input/Output:
        `None`
        """
        contours, _ = cv2.findContours(self.__processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=sorting_function_contours, reverse=True)

        good_contours = []
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
        """
        Makes a prediction of a contour that is most likely the car.
        
        Input/Output:
        `None`
        """
        if self.__current_contours == 0:
            return
        else:
            prediction = ()
            for contour in self.__current_contours:
                x, y, w, h = cv2.boundingRect(contour)
                distance = calculate_distance(self.__bbox, [x, y, w, h])
                if self.__previous_contours == [] or len(self.__previous_contours) >= len(self.__current_contours):
                    if prediction == () or distance < prediction[0]:
                        prediction = (distance, [x, y, w, h])
                else:
                    if distance > 100:
                        continue
                    if prediction == () or distance > prediction[0]:
                        prediction = (distance, [x, y, w, h])

            if prediction:
                self.__previous_contours = self.__current_contours
                self.__bbox = prediction[1]

    def __visualize_contours(self) -> None:
        """
        Visualizes the contours with their bounding boxes on the processed frame.
        
        Input/Output:
        `None`
        """
        self.__visualized_frame = self.__frame.copy()

        # Draws green boxes around each possibility.
        for contour in self.__current_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(self.__visualized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draws a white box around the car
        x, y, w, h = self.__bbox
        cv2.rectangle(self.__visualized_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    def __write_frame_to_video(self) -> None:
        """
        Writes the last visualized frame to the Video.
        
        Input/Output:
        `None`
        """
        if self.__RECORD_VIDEO:
            self.__OUTPUT_VIDEO.write(self.__visualized_frame)

    def __show_frame(self) -> None:
        """
        Shows the last visualized frame and allows to exit application with q.
        
        Inout/Output:
        `None`
        """
        if self.__SHOW_TRACKING_VIEW:
            cv2.imshow("Car Tracking", self.__visualized_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt("User pressed 'q' to stop the visualisation.")

    def __send_bbox_coordinates(self):
        """
        Sends the middle coordinates of the car using pynng.
        
        Input/Output:
        `None`
        """
        middle = (self.__bbox[0] + self.__bbox[2] / 2,
                  self.__bbox[1] + self.__bbox[3] / 2)
        json_str = dumps(middle)
        self.__ADDRESS_SENDER.send(json_str.encode("utf-8"))

    # Main Function
    def main(self):
        """
        Is an infinite loop that goes through the Camera Stream. !DEBUGGING ONLY: Uses video instead of camera stream.!
        
        Input/Output:
        `None`
        """
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


if __name__ == "__main__":
    vt = VehicleTracker(video_path="C:\\Users\\VWMFM88\\Downloads\\drive_990p.h265")
    vt.main()
