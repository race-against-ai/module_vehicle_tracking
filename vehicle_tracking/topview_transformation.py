from typing import Tuple
import numpy as np
import cv2


class TransformationPoint:
    def __init__(self, camera_image_point: Tuple[int, int], topview_image_point: Tuple[int, int],
                 world_coordinate_system_point: Tuple[float, float]) -> None:
        self.camera_image_point = camera_image_point
        self.topview_image_point = topview_image_point
        self.world_coordinate_system_point = world_coordinate_system_point


class TopViewTransformation:
    def __init__(self, top_left_point: TransformationPoint, top_right_point: TransformationPoint,
                 bottom_left_point: TransformationPoint, bottom_right_point: TransformationPoint) -> None:
        self.top_left_point = top_left_point
        self.top_right_point = top_right_point
        self.bottom_left_point = bottom_left_point
        self.bottom_right_point = bottom_right_point

        self.camera_image_pts = np.array([list(top_left_point.camera_image_point),
                                          list(bottom_left_point.camera_image_point),
                                          list(bottom_right_point.camera_image_point),
                                          list(top_right_point.camera_image_point)],
                                         dtype=np.float32)

        self.topview_image_pts = np.array([list(top_left_point.topview_image_point),
                                           list(bottom_left_point.topview_image_point),
                                           list(bottom_right_point.topview_image_point),
                                           list(top_right_point.topview_image_point)],
                                          dtype=np.float32)

        self.world_coordinate_system_pts = np.array([list(top_left_point.world_coordinate_system_point),
                                                     list(bottom_left_point.world_coordinate_system_point),
                                                     list(bottom_right_point.world_coordinate_system_point),
                                                     list(top_right_point.world_coordinate_system_point)],
                                                    dtype=np.float32)

        self.camera_to_topview_transformation_matrix = cv2.getPerspectiveTransform(self.camera_image_pts,
                                                                                   self.topview_image_pts)

        self.camera_to_world_transformation_matrix = cv2.getPerspectiveTransform(self.camera_image_pts,
                                                                                 self.world_coordinate_system_pts)

    def transform_camera_image_to_topview_image(self, camera_image: np.ndarray):
        return cv2.warpPerspective(camera_image, self.camera_to_topview_transformation_matrix,
                                   camera_image.shape[:2][::-1])

    def transform_camera_point_to_world_coordinate(self, camera_point: Tuple[int, int]) -> Tuple[float, float]:
        point = np.array([list(camera_point)], dtype='float32')
        return cv2.perspectiveTransform(point[None, :, :], self.camera_to_world_transformation_matrix)
