"""A virtual camera to test the drawing of objects on a frame."""
# Copyright (C) 2023, NG:ITL

from time import sleep

import numpy as np
import cv2


# Constant Functions
def get_path_pixels(path: list[tuple[int, int]] | list[list[int]]) -> list[tuple[int, int]]:
    """Get all pixels in a path.

    Args:
        path (list[tuple[int, int]]): The corners of the path.

    Returns:
        list[tuple[int, int]]: The pixels in the path.
    """

    def get_line_pixels(x_start: int, y_start: int, x_end: int, y_end: int) -> list[tuple[int, int]]:
        delta_x = abs(x_end - x_start)
        delta_y = abs(y_end - y_start)
        step_x = 1 if x_start < x_end else -1
        step_y = 1 if y_start < y_end else -1
        error = delta_x - delta_y

        pixels = []
        while True:
            pixels.append((x_start, y_start))

            if x_start == x_end and y_start == y_end:
                break

            double_error = 2 * error
            if double_error > -delta_y:
                error -= delta_y
                x_start += step_x
            if double_error < delta_x:
                error += delta_x
                y_start += step_y

        return pixels

    pixels: list[tuple[int, int]] = []
    num_points = len(path)
    for i in range(num_points):
        start_x, start_y = path[i]
        end_x, end_y = path[(i + 1) % num_points]
        pixels.extend(get_line_pixels(start_x, start_y, end_x, end_y))

    formatted_pixels: list[tuple[int, int]] = [(x, y) for x, y in pixels]
    return formatted_pixels


# Class
class ToDrawObject:
    """A class to represent an object to draw by the virtual camera."""

    def __init__(
        self,
        color: tuple[int, int, int],
        shape: list[tuple[int, int]],
        speed: float,
        path: list[tuple[int, int]],
    ) -> None:
        """Create a ToDrawObject.

        Args:
            color (tuple[int, int, int]): The color of the object.
            shape (list[tuple[int, int]]): The shape of the object.
            speed (float): The speed of the object. (calculated and rounded down for index of path)
            path (list[tuple[int, int]]): The path of the object.
        """
        self.color = color
        self.shape = shape
        self.speed = speed
        self.path = path
        self.iteration = 0
        self.__unrounded_iteration: float = 0
        x_points: list[int] = []
        y_points: list[int] = []
        for point in self.shape:
            x_points.append(point[0])
            y_points.append(point[1])
        x_points.sort()
        y_points.sort()
        biggest_x = x_points[-1]
        biggest_y = y_points[-1]
        self.centroid = (biggest_x // 2, biggest_y // 2)

    def get_next_position(self) -> tuple[int, int]:
        """Get the next position of the object.

        Returns:
            tuple[int, int]: The next position of the object.
        """
        position = self.path[self.iteration % len(self.path)]
        self.__unrounded_iteration += self.speed
        self.iteration = int(self.__unrounded_iteration)
        return position


class VirtualCamera:
    """A class that draws objects on a frame to allow easy testing."""

    def __init__(self, to_draw_objects: list[ToDrawObject], frame_rate: int) -> None:
        """Create a VirtualCamera.

        Args:
            to_draw_objects (list[ToDrawObject]): The objects to draw on the frame.
            frame_rate (int): The frame rate of the camera.
        """
        self.frame_size: tuple[int, int, int] = (990, 1332, 3)
        self.to_draw_objects = to_draw_objects
        self.__time_to_sleep = 1 / frame_rate
        self.__current_frame: np.ndarray = np.ndarray(self.frame_size, dtype=np.uint8)
        self.__next_frame = self.__current_frame.copy()
        for _ in range(2):
            self.__generate_next_frame()

    def __generate_next_frame(self) -> None:
        """Generate the next frame and draw all objects on it."""
        self.__current_frame = self.__next_frame.copy()
        self.__next_frame = np.zeros(self.frame_size, dtype=np.uint8)
        for to_draw_object in self.to_draw_objects:
            x_pos, y_pos = to_draw_object.get_next_position()
            cv2.fillPoly(
                self.__next_frame,
                [np.array(to_draw_object.shape)],
                to_draw_object.color,
                offset=(x_pos - to_draw_object.centroid[0], y_pos - to_draw_object.centroid[1]),
            )

    def read_new_frame(self) -> np.ndarray:
        """Read a new frame from the camera.

        Returns:
            np.ndarray: The new frame.
        """
        current = self.__current_frame.copy()
        self.__generate_next_frame()
        sleep(self.__time_to_sleep)
        return current


if __name__ == "__main__":
    car1 = ToDrawObject(
        (9, 103, 246),
        [(0, 0), (20, 0), (20, 20), (0, 20)],
        1,
        get_path_pixels([(100, 100), (200, 100), (200, 200), (100, 200)]),
    )
    car2 = ToDrawObject(
        (0, 255, 0),
        [(0, 0), (20, 0), (20, 20), (0, 20)],
        2,
        get_path_pixels([(100, 100), (200, 100), (200, 200), (100, 200)]),
    )
    wall = ToDrawObject((255, 255, 255), [(0, 0), (20, 0), (20, 20), (0, 20)], 0, [(0, 0)])

    cam = VirtualCamera([car1, car2, wall], 60)
    while True:
        frame = cam.read_new_frame()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
