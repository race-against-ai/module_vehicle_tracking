"""A module that contains classes for getting images from different sources."""
# Copyright (C) 2023, NG:ITL

from pathlib import Path
from time import sleep

from pynng import Sub0
import numpy as np
import cv2


class VideoFileSource:
    """An image source which reads images from a file.

    Args:
        video_file_filepath (Path): The path of the video to be read from.
        frame_rate (int): The frame rate of the video.
    """

    def __init__(self, video_file_filepath: Path, frame_rate: int) -> None:
        self.__time_to_sleep = 1 / frame_rate
        self.__video_capture = cv2.VideoCapture(str(video_file_filepath))  # type: ignore[call-arg]
        self.frame_size = self.read_new_frame().shape

    def read_new_frame(self) -> np.ndarray:
        """This function reads a frame and returns the image.

        Raises:
            IndexError: If the file is not found or if it is empty.

        Returns:
            np.ndarray: The frame to be returned.
        """
        success, frame = self.__video_capture.read()
        if not success:
            raise IndexError("The next frame could not be read. The frame is empty.")
        sleep(self.__time_to_sleep)
        return frame


class CameraStreamSource:
    """An image source which reads images from a camera stream.

    Args:
        address (str): The pynng address to connect to the camera stream.
    """

    def __init__(self, address: str) -> None:
        self.frame_size = (990, 1332, 3)
        self.__frame_receiver = Sub0()
        self.__frame_receiver.subscribe("")
        self.__frame_receiver.dial(address)

    def read_new_frame(self) -> np.ndarray:
        """This function reads a frame and returns the image.

        Returns:
            np.ndarray: The frame to be returned.
        """
        frame_bytes = self.__frame_receiver.recv()
        frame = np.frombuffer(frame_bytes, np.uint8)
        frame = frame.reshape(self.frame_size)
        return frame
