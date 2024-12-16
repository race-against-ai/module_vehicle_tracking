"""This module contains the TopViewTransformation class which is used to transform a point from the camera image to coordinates in meters in the real world.ye"""
# Copyright (C) 2023, NG:ITL
from typing import NamedTuple
import numpy as np
import cv2


class TransformationPoint(NamedTuple):
    """A point in the camera image and the corresponding point in the world coordinate system.

    Args:
        camera_image_point (tuple[int, int]): The point in the camera image.
        world_coordinate_point (tuple[float, float]): The corresponding point in the real world."""

    camera_image_point: tuple[int, int]
    world_coordinate_point: tuple[float, float]


class TopViewTransformation:
    """A class that transforms a point from the camera image to coordinates in meters in the real world."""

    def __init__(
        self,
        top_left_point: TransformationPoint,
        top_right_point: TransformationPoint,
        bottom_left_point: TransformationPoint,
        bottom_right_point: TransformationPoint,
    ) -> None:
        """Constructor for the TopViewTransformation class.

        Args:
            top_left_point (TransformationPoint): The top left transformation point.
            top_right_point (TransformationPoint): The top right transformation point.
            bottom_left_point (TransformationPoint): The bottom left transformation point.
            bottom_right_point (TransformationPoint): The bottom right transformation point.
        """
        camera_image_pts = np.array(
            [
                list(top_left_point.camera_image_point),
                list(bottom_left_point.camera_image_point),
                list(bottom_right_point.camera_image_point),
                list(top_right_point.camera_image_point),
            ],
            dtype=np.float32,
        )

        world_coordinate_system_pts = np.array(
            [
                list(top_left_point.world_coordinate_point),
                list(bottom_left_point.world_coordinate_point),
                list(bottom_right_point.world_coordinate_point),
                list(top_right_point.world_coordinate_point),
            ],
            dtype=np.float32,
        )

        self.camera_to_world_transformation_matrix = cv2.getPerspectiveTransform(
            camera_image_pts, world_coordinate_system_pts
        )

    def transform_camera_point_to_world_coordinate(self, camera_point: tuple[int, int]) -> tuple[float, float]:
        """Transforms a point from the camera image to coordinates in meters in the real world."""
        point = np.array([list(camera_point)], dtype=np.float32)
        return cv2.perspectiveTransform(point[None, :, :], self.camera_to_world_transformation_matrix)
