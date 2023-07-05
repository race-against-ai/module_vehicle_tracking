# Copyright (C) 2022, NG:ITL

import json
import pynng
import cv2
import numpy as np

address_pub_coordinates = 'ipc:///tmp/RAAI/vehicle_coordinates_fallback.ipc'
address_rec_frames = 'ipc:///tmp/RAAI/camera_frame.ipc'


def read_config(config_file_path: str) -> dict:
    with open(config_file_path, 'r') as file:
        return json.load(file)


class Tracker:
    def __init__(self, config_file_path='../config.json', p_use_camera_stream: bool = True, video_path: str = ''):

        self.__checkpoints = []
        self.__use_camera_stream = p_use_camera_stream
        self.__define_cap(video_path)

        # getting checkpoint positions from config file
        self.__config = read_config(config_file_path)
        self.__number_of_checkpoints = len(self.__config["checkpoints"])
        self.__checkpoint_list = self.__config["checkpoints"]

        for i in range(self.__number_of_checkpoints):
            if i == 0:
                self.__checkpoints.append(FinishLineCheckpoint(self.__checkpoint_list[i], self.__number_of_checkpoints))
            else:
                self.__checkpoints.append(SectorLineCheckpoint(self.__checkpoint_list[i], i))

        # setting up a pynng socket
        self.__pub = pynng.Pub0()
        self.__pub.listen(address_pub_coordinates)

    def __define_cap(self, video_path: str) -> None:
        """
        Defines the video cap.

        Input:
        video_path: str -> if use_camera_stream = True then enter the video path else leave it as a blank string. (Debugging)

        Output:
        None
        """
        if self.__use_camera_stream:
            self.__define_image_receiver()
            self.__video_size = (480, 640, 3)
            self.__read_new_frame()
        else:
            self.__video_cap = cv2.VideoCapture(video_path)
            success, self.__frame = self.__video_cap.read()
            self.__video_size = self.__frame.shape
            if not success:
                raise FileNotFoundError(
                    'Could not read the first frame of the video. Please try again and validate the video path.')

            self.__video_size = self.__frame.shape[:2][::-1]

    def __read_new_frame(self) -> None:
        """
        Reads the next frame in the video.

        Input/Output:
        None
        """
        if self.__use_camera_stream:
            frame_bytes = self.__sub.recv()
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            self.__frame = frame.reshape(self.__video_size)
        else:
            success, self.__frame = self.__video_cap.read()
            if not success:
                raise IndexError('Frame could not be read. Video probably ended.')

    def __define_image_receiver(self) -> None:
        """
        Defines the image receiver, so it can receive the camera's images.

        Input/Output:
        None
        """
        self.__sub = pynng.Sub0()
        self.__sub.subscribe('')
        self.__sub.dial(address_rec_frames)

    def __show_frame(self) -> None:
        """
        Shows the last visualized frame and allows to exit application with q.

        Inout/Output:
        None
        """
        cv2.imshow('Car Tracking', self.__frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt('User pressed "q" to stop the visualisation.')

    def checkpoint_check(self, camera_image):
        for checkpoint in self.__checkpoints:
            if isinstance(checkpoint, FinishLineCheckpoint):
                if checkpoint.check(camera_image):
                    checkpoint.send(self.__pub)
                    for checkpoint_help in self.__checkpoints:
                        checkpoint_help.set_crossed(False)
            else:
                if checkpoint.check(camera_image):
                    checkpoint.send(self.__pub)

    def main(self):
        """
        Is an infinite loop that goes through the Camera Stream. !DEBUGGING ONLY: Uses video instead of camera stream.!

        Input/Output:
        None
        """
        while True:
            self.__read_new_frame()
            self.checkpoint_check(self.__frame)
            self.__show_frame()


class Checkpoint:
    def __init__(self, checkpoint: dict, p_num: int) -> None:
        self.__x1 = checkpoint["x1"]
        self.__y1 = checkpoint["y1"]
        self.__x2 = checkpoint["x2"]
        self.__y2 = checkpoint["y2"]
        self.__crossed = False
        self.__num = p_num

    def check_line(self, picture, i=0):
        # checks if the car drives through the given Pixels
        pixel_cap = 0
        for y in range(self.__y1 + i, self.__y2 + i):
            for x in range(self.__x1, self.__x2):
                if picture.item(y, x, 2) in range(100, 250) and picture.item(y, x, 1) < 100 > picture.item(y, x, 0):
                    pixel_cap += 1
        if pixel_cap > 30:
            self.__crossed = True
            return True
        return False

    def send(self, p_pub):
        msg = self.__num
        print(msg)
        p_pub.send(msg.to_bytes(4, 'big'))

    def get_crossed(self) -> bool:
        return self.__crossed

    def set_crossed(self, x: bool) -> None:
        self.__crossed = x


class FinishLineCheckpoint(Checkpoint):
    def __init__(self, checkpoint: dict, p_num: int) -> None:
        super().__init__(checkpoint, p_num)

    def check(self, p_image) -> bool:
        if self.get_crossed() is True:
            if self.check_line(p_image):
                return True
        else:
            self.check_line(p_image, 60)
        return False


class SectorLineCheckpoint(Checkpoint):
    def __init__(self, checkpoint: dict, p_num: int) -> None:
        super().__init__(checkpoint, p_num)

    def check(self, p_image) -> bool:
        if self.get_crossed() is False:
            if self.check_line(p_image):
                return True
        return False


if __name__ == '__main__':
    vt = Tracker(p_use_camera_stream=False, video_path='../../../../TestVideo/drive_990p.h265')
    vt.main()
