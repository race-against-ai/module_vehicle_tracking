"""Provides a class for transforming points between image and world coordinates."""
# Copyright (C) 2023, NG:ITL

import cv2
import numpy as np


class TopviewTransformation:
    """Transforms a camera point to a world coordinate and vice versa."""

    def __init__(self) -> None:
        self.__image_points: dict[str, tuple[int, int]] = {
            "top_left": (0, 0),
            "top_right": (0, 0),
            "bottom_left": (0, 0),
            "bottom_right": (0, 0),
        }
        self.__world_points: dict[str, tuple[float, float]] = {
            "top_left": (0, 0),
            "top_right": (0, 0),
            "bottom_left": (0, 0),
            "bottom_right": (0, 0),
        }
        self.__image_to_world_matrix = cv2.getPerspectiveTransform(  # type: ignore[type-var]
            np.array(list(self.__image_points.values()), dtype=np.float32),
            np.array(list(self.__world_points.values()), dtype=np.float32),
        )
        self.__world_to_image_matrix = cv2.getPerspectiveTransform(  # type: ignore[type-var]
            np.array(list(self.__world_points.values()), dtype=np.float32),
            np.array(list(self.__image_points.values()), dtype=np.float32),
        )

    def set_transformation_point(
        self, point_name: str, image_coords: tuple[int, int], world_coords: tuple[float, float]
    ) -> None:
        """Sets a transformation point.

        Args:
            point_name (str): The name of the transformation point.
            image_coords (tuple[int, int]): The image coordinates of the transformation point.
            world_coords (tuple[float, float]): The world coordinates of the transformation point.
        """
        self.__image_points[point_name] = image_coords
        self.__world_points[point_name] = world_coords

        image_points = list(self.__image_points.values())
        world_points = list(self.__world_points.values())
        self.__image_to_world_matrix = cv2.getPerspectiveTransform(  # type: ignore[type-var]
            np.array(image_points, dtype=np.float32), np.array(world_points, dtype=np.float32)
        )
        self.__world_to_image_matrix = cv2.getPerspectiveTransform(  # type: ignore[type-var]
            np.array(world_points, dtype=np.float32), np.array(image_points, dtype=np.float32)
        )

    def image_to_world_transform(self, point: tuple[int, int]) -> tuple[float, float]:
        """Transforms an image point to a world point.

        Args:
            point (tuple[int, int]): The image point to transform.

        Returns:
            tuple[float, float]: The world point.
        """
        points = cv2.perspectiveTransform(np.array([[point]], dtype=np.float32), self.__image_to_world_matrix)[0][0]  # type: ignore[type-var]
        return (round(float(points[0]), 3), round(float(points[1]), 3))

    def world_to_image_transform(self, point: tuple[float, float]) -> tuple[int, int]:
        """Transforms a world point to an image point.

        Args:
            point (tuple[float, float]): The world point to transform.

        Returns:
            tuple[int, int]: The image point.
        """
        points = cv2.perspectiveTransform(np.array([[point]], dtype=np.float32), self.__world_to_image_matrix)[0][0]  # type: ignore[type-var]
        return (int(points[0]), int(points[1]))
