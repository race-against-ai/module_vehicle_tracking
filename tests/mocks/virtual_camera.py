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

    def get_line_pixels(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        pixels = []
        while True:
            pixels.append((x0, y0))

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return pixels

    pixels: list[tuple[int, int]] = []
    num_points = len(path)
    for i in range(num_points):
        x0, y0 = path[i]
        x1, y1 = path[(i + 1) % num_points]
        pixels.extend(get_line_pixels(x0, y0, x1, y1))

    formatted_pixels: list[tuple[int, int]] = [(x, y) for x, y in pixels]
    return formatted_pixels


# Class
class ToDrawObject:
    def __init__(
        self,
        color: tuple[int, int, int],
        shape: list[tuple[int, int]],
        speed: float,
        path: list[tuple[int, int]],
    ) -> None:
        self.color = color
        self.shape = shape
        self.speed = speed
        self.path = path
        self.iteration = 0
        self.__unrounded_iteration = 0
    
    def get_next_position(self) -> tuple[int, int]:
        """Get the next position of the object.

        Returns:
            `tuple[int, int]`: The next position of the object.
        """
        position = self.path[self.iteration % len(self.path)]
        self.__unrounded_iteration += self.speed
        self.iteration = int(self.__unrounded_iteration)
        return position


class VirtualCamera:
    def __init__(self, to_draw_objects: list[ToDrawObject], frame_rate: int) -> None:
        """Create a virtual camera mimicking the behavior of a real camera.

        Args:
            `to_draw_objects (list[ToDrawObject])`: A list of objects to draw on the camera. The first object bottom, last object top.
            `frame_rate (int)`: The framerate of the virtual camera.
        """
        self.frame_size: tuple[int, int, int] = (990, 1332, 3)
        self.to_draw_objects = to_draw_objects
        self.__time_to_sleep = 1 / frame_rate
        self.__current_frame = np.ndarray(self.frame_size, dtype=np.uint8)
        self.__next_frame = self.__current_frame.copy()
        for _ in range(2):
            self.__generate_next_frame()
    
    def __generate_next_frame(self) -> None:
        """Generate the next frame and draw all objects on it."""
        self.__current_frame = self.__next_frame.copy()
        self.__next_frame = np.zeros(self.frame_size, dtype=np.uint8)
        for to_draw_object in self.to_draw_objects:
            pX, pY = to_draw_object.get_next_position()
            shape_points = to_draw_object.shape.copy()
            x_points, y_points = [p[0] for p in shape_points], [p[1] for p in shape_points]
            x_points.sort(reverse=True)
            y_points.sort(reverse=True)
            centroid = (x_points[0] // 2, y_points[0] // 2)
            cv2.fillPoly(self.__next_frame, [np.array(shape_points)], to_draw_object.color, offset=(pX - centroid[0], pY - centroid[1]))
            cv2.circle(self.__next_frame, (pX, pY), 3, (0, 0, 255), -1)
    
    def read_new_frame(self) -> np.ndarray:
        """Read a new frame from the camera.

        Returns:
            `np.ndarray`: The new frame.
        """
        current = self.__current_frame.copy()
        self.__generate_next_frame()
        sleep(self.__time_to_sleep)
        return current


if __name__ == "__main__":
    car1 = ToDrawObject((9, 103, 246), [(0, 0), (20, 0), (20, 20), (0, 20)], 1, get_path_pixels([(100, 100), (200, 100), (200, 200), (100, 200)]))
    car2 = ToDrawObject((0, 255, 0), [(0, 0), (20, 0), (20, 20), (0, 20)], 2, get_path_pixels([(100, 100), (200, 100), (200, 200), (100, 200)]))
    wall = ToDrawObject((255, 255, 255), [(0, 0), (20, 0), (20, 20), (0, 20)], 0, [(0, 0)])
    
    cam = VirtualCamera([car1, car2, wall], 60)
    while True:
        frame = cam.read_new_frame()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
