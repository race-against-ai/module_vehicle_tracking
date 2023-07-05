# Imports
from typing import List, Tuple
from pynng import Sub0
from json import dump
from os import path
import numpy as np
import cv2


# Constants
FRAME_RECEIVE_LINK = "ipc:///tmp/RAAI/camera_frame.ipc"
CURRENT_DIR = path.dirname(path.abspath(__file__))


# Classes
class ROIDefiner:
    # Initialization
    def __init__(self, video_path: str = "") -> None:
        """The main function for initializing the ROI Definer.

        Args:
            `video_path (str, optional)`: If not empty then it uses the video instead of camera stream. Defaults to "".
        """
        self.__roi_points: List[Tuple[int, int]] = []
        self.__VIDEO_PATH = video_path
        
        self.__define_cap()
        self.__define_windows()
        
    def __define_cap(self) -> None:
        """Defines the `cv2.VideoCapture()`

        Raises:
            FileNotFoundError: If the `video_path` in the `__init__` doesn't exist or is empty.
        """
        if self.__VIDEO_PATH == "":
            self.__define_image_receiver()
            self.__RESHAPE_VIDEO_SIZE = (480, 640, 3)
            self.__read_new_frame()
        else:
            self.__VIDEO_CAP = cv2.VideoCapture(self.__VIDEO_PATH)
            success, self.__frame = self.__VIDEO_CAP.read()
            if not success:
                raise FileNotFoundError("Could not read the first frame of the video.")
        
    def __define_image_receiver(self) -> None:
        """Defines the image receiver, so it can receive the cameras images.
        """
        self.__FRAME_RECEIVER = Sub0()
        self.__FRAME_RECEIVER.subscribe("")
        self.__FRAME_RECEIVER.dial(FRAME_RECEIVE_LINK)
    
    def __define_windows(self) -> None:
        """Defines the windows to be used by `cv2.imshow()`
        """
        cv2.namedWindow("Point Drawer")
        cv2.namedWindow("Region Of Interest")
        
        cv2.setMouseCallback("Point Drawer", self.__mouse_event_handler)

    # Mouse Handler
    def __mouse_event_handler(self, event: int, x: int, y: int, _flags, _params) -> None:
        """A function that handles the mouse events sent from `cv2.setMouseCallback()`

        Args:
            event (int): The event id of the mouse.
            x (int): The x position of the mouse.
            y (int): The y position of the mouse.
            _flags : Not used/Placeholder.
            _params : Not used/Placeholder.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__roi_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.__roi_points.pop()
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.__roi_points = []

    # Helper Functions
    def __read_new_frame(self) -> None:
        """Reads the next frame in the video.

        Raises:
            IndexError: Is raised if the video has ended or the next frame is empty.
        """
        if self.__VIDEO_PATH == "":
            frame_bytes = self.__FRAME_RECEIVER.recv()
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            self.__frame = frame.reshape(self.__RESHAPE_VIDEO_SIZE)
        else:
            success, self.__frame = self.__VIDEO_CAP.read()
            if not success:
                raise IndexError("Frame could not be read. Video probably ended.")
    
    def __draw_on_frame(self) -> None:
        """Draws lines connecting the ROI points. Draws ROI (if >= 3 points).
        """
        self.__lined_frame = self.__frame.copy()
        if len(self.__roi_points) >= 1:
            self.__lined_frame = cv2.polylines(self.__lined_frame, [np.array(self.__roi_points)], True,
                                               (255, 255, 255), 2)
        if len(self.__roi_points) >= 3:
            np_points = np.array(self.__roi_points)
            mask = np.zeros_like(self.__frame)
            cv2.fillPoly(mask, [np_points], (255, 255, 255))
            self.__updated_frame = cv2.bitwise_and(self.__frame, mask)
        else:
            self.__updated_frame = np.zeros_like(self.__frame)
    
    def __show_images(self) -> None:
        """Uses `cv2.imshow()` to send the frame to the `cv2.namedWindow`.
        """
        cv2.imshow("Point Drawer", self.__lined_frame)
        cv2.imshow("Region Of Interest", self.__updated_frame)
        
    def __check_to_close(self, force_quit: bool = False) -> None:
        """Uses `cv2.waitKey()` to see if the user presses 'q' to exit the program.
        Can also be called with `force_quit = True` to quit the program and save the ROI.

        Args:
            force_quit (bool, optional): If set to true exits the program and saves the ROI. Defaults to False.

        Raises:
            KeyboardInterrupt: Triggers the interrupt to exit the program.
        """
        if force_quit or cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            self.__save_roi()
            raise KeyboardInterrupt("User has interrupted the program.")

    def __save_roi(self) -> None:
        """Saves the ROI as `region_of_interest.json`.
        """
        if len(self.__roi_points) >= 3:
            with open(path.join(CURRENT_DIR, "region_of_interest.json"), "x") as f:
                dump(self.__roi_points, f)
        else:
            print("Not enough points set.")

    # Main Functions
    def main(self) -> None:
        """Is an infinite loop that goes through the camera stream/video.
        """
        while True:
            self.__read_new_frame()
            self.__draw_on_frame()
            self.__show_images()
            self.__check_to_close()


if __name__ == "__main__":
    newROIDefiner = ROIDefiner()
    newROIDefiner.main()
