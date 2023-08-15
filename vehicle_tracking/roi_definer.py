# Copyright (C) 2023, NG:ITl
from image_sources import CameraStreamSource, VideoFileSource
from json import dump, load
from pathlib import Path
import numpy as np
import cv2


# Constants
CURRENT_DIR = Path(__file__).parent


# Classes
class ROIDefiner:
    def __init__(self, image_source: CameraStreamSource | VideoFileSource) -> None:
        """ROI Definer class.

        Args:
            image_source (CameraStreamSource | VideoFileSource): Image source to get frames from.
        """
        self.__image_source = image_source
        self.__roi_points: list[tuple[int, int]] = []

        cv2.namedWindow("Point Drawer")
        cv2.namedWindow("Region Of Interest")
        cv2.setMouseCallback("Point Drawer", self.__mouse_event_handler)

    def __mouse_event_handler(self, event: int, x: int, y: int, _flags, _params) -> None:
        """Handles mouse events.

        Args:
            event (int): The cv2.EVENT_* event.
            x (int): The x position of the cursor.
            y (int): The y position of the cursor.
            _flags (Any): Unused.
            _params (Any): Unused.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__roi_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.__roi_points.pop()
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.__roi_points = []

    def __process_new_frame(self) -> None:
        """Processes a new frame from the image source."""
        frame = self.__image_source.read_new_frame()
        self.__point_drawer_frame = frame.copy()
        if len(self.__roi_points) >= 1:
            self.__point_drawer_frame = cv2.polylines(frame, [np.array(self.__roi_points)], True, (255, 255, 255), 2)
        self.__roi_frame = np.zeros_like(frame)
        if len(self.__roi_points) >= 3:
            self.__roi_frame = cv2.fillPoly(self.__roi_frame, [np.array(self.__roi_points)], (255, 255, 255))
            self.__roi_frame = cv2.bitwise_and(frame, self.__roi_frame)

    def __show_frames(self) -> None:
        """Shows the frame on the screen."""
        cv2.imshow("Point Drawer", self.__point_drawer_frame)
        cv2.imshow("Region Of Interest", self.__roi_frame)

    def __check_to_close(self) -> None:
        """Checks if it should close the program.

        Raises:
            KeyboardInterrupt: If the user pressed 'q' to close the program.
        """
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            self.__save_roi()
            exit(-1)

    def __save_roi(self) -> None:
        """Saves the ROI to the config file."""
        if len(self.__roi_points) < 3:
            raise ValueError("ROI not saved. You need at least 3 points to define a region of interest.")
        with open(CURRENT_DIR.parent / "vehicle_tracking_config.json", "r") as f:
            config = load(f)
            config["roi_points"] = self.__roi_points
        with open(CURRENT_DIR.parent / "vehicle_tracking_config.json", "w") as f:
            dump(config, f, indent=4)

    def run(self) -> None:
        """Runs the ROI definer."""
        while True:
            self.__process_new_frame()
            self.__show_frames()
            self.__check_to_close()


if __name__ == "__main__":
    image_source = VideoFileSource(CURRENT_DIR.parent / "resources/test_video_1.h265", 60)
    roi_definer = ROIDefiner(image_source)
    roi_definer.run()
