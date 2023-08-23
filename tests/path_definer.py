# Copyright (C) 2023, NG:ITL
from pathlib import Path
from json import dumps
import numpy as np
import pyperclip
import cv2


# Constants
CURRENT_DIR = Path(__file__).parent


# Classes
class PathDefiner:
    def __init__(self) -> None:
        """A class for defining a path for testing purposes. The path will, when exited with q, be saved to the users clipboard."""
        self.__frame_size: tuple[int, int, int] = (990, 1332, 3)
        self.__points: list[tuple[int, int]] = []
        cv2.namedWindow("Path Drawer")
        cv2.setMouseCallback("Path Drawer", self.__mouse_event_handler)

    def __mouse_event_handler(self, event: int, x: int, y: int, _flags, _params) -> None:
        """The mouse event handler for the cv2.setMouseCallback().

        Args:
            event (int): The event type.
            x (int): The x coordinate of the mouse.
            y (int): The y coordinate of the mouse.
            _flags (Any): Unused.
            _params (Any): Unused.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.__points.pop()
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.__points = []

    def __read_new_frame(self) -> None:
        """Reads a new frame from the camera."""
        self.__frame = np.zeros(self.__frame_size, np.uint8)

    def __draw_path(self) -> None:
        """Draws the path on the frame."""
        cv2.polylines(self.__frame, [np.array(self.__points)], True, (255, 255, 255), 2)

    def __show_frame(self) -> None:
        """Shows the frame on the screen."""
        cv2.imshow("Path Drawer", self.__frame)

    def __check_to_close(self, force_quit: bool = False) -> None:
        """Checks if the window should be closed and the path saved.

        Args:
            force_quit (bool, optional): If set to true it will exit without user input. Defaults to False.
        """
        if cv2.waitKey(1) & 0xFF == ord("q") or force_quit:
            cv2.destroyAllWindows()
            self.__save_path()
            exit(-1)

    def __save_path(self) -> None:
        """Saves the path to the clipboard of the user."""
        dumped = dumps(self.__points, indent=4)
        replaced = "[\n\t" + dumped[1:-1].replace("[", "(").replace("]", ")").replace("), (", "),\n\t(") + "\n]"
        pyperclip.copy(replaced)

    def run(self) -> None:
        """Runs the path definer."""
        while True:
            self.__read_new_frame()
            self.__draw_path()
            self.__show_frame()
            self.__check_to_close()


if __name__ == "__main__":
    definer = PathDefiner()
    definer.run()
