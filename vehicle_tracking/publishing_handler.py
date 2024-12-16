"""Handles all the communication between clients."""
# Copyright (C) 2023, NG:ITL

from json import load, loads, dumps
from threading import Thread, Event
from pathlib import Path
from math import floor

from jsonschema import validate, ValidationError
from pynng import Rep0, Pub0, Req0, Closed


CURRENT_DIR = Path(__file__).parent


class _PositionSender:
    """Sends the position of the tracked box over pynng.

    Args:
        vehicle_tracking: The vehicle tracker object.
        address (str): The address to send the position to.
        topics (dict[str, str]): The topics to send the position to.
    """

    def __init__(self, vehicle_tracking, address: str, topics: dict[str, str]) -> None:
        self.__position_sender = Pub0(listen=address, send_timeout=1000)
        self.__tracker = vehicle_tracking
        self.__image_topic = topics["coords_image"]
        self.__real_topic = topics["coords_world"]

    def send_position(self) -> None:
        """Sends the image and real world coordinates of the tracked box over pynng."""
        bbox = self.__tracker.bbox
        image_coordinate = (bbox[0] + floor(bbox[2] / 2), bbox[1] + floor(bbox[3] / 2))

        image_pos_string = self.__image_topic + " " + dumps(image_coordinate)

        real_world_coords = self.__tracker.topview_transformation.image_to_world_transform(image_coordinate)
        real_pos_string = self.__real_topic + " " + dumps(real_world_coords)

        self.__position_sender.send(image_pos_string.encode("utf-8"))
        self.__position_sender.send(real_pos_string.encode("utf-8"))


class _ProcessedFrameSender:
    """Sends the processed frame over pynng.

    Args:
        vehicle_tracking: The vehicle tracker object.
        address (str): The address to send the frame to.
    """

    def __init__(self, vehicle_tracking, address: str) -> None:
        self.__tracker = vehicle_tracking
        self.__processed_frame_sender = Pub0(listen=address, send_timeout=1000)

    def send_processed_frame(self) -> None:
        """Sends the processed frame over pynng."""
        frame = self.__tracker.visualized_frame.tobytes()
        self.__processed_frame_sender.send(frame)


class _RequestServer:
    """Handles all incoming requests for the vehicle tracker.

    Args:
        vehicle_tracking: The vehicle tracker object.
        address (str): The address to listen for requests on.
        stop_flag (Event): The flag to stop awaiting requests.
    """

    def __init__(self, vehicle_tracking, address: str, stop_flag: Event) -> None:
        self.__stop_awaiting_request = stop_flag
        self.__address = address

        schemas_to_open = ["request", "response"]
        self.schemas: dict[str, dict] = {}
        for schema in schemas_to_open:
            with open(CURRENT_DIR / f"schemas/{schema}.json", "r", encoding="utf-8") as f:
                self.schemas[schema] = load(f)

        self.__request_handler = Rep0(listen=address)
        self.__tracker = vehicle_tracking
        self.__worker_thread = Thread(target=self.__worker)
        self.__worker_thread.start()
        self.exited = False

    def stop_server(self) -> None:
        """Stops the request server."""
        print("Stopping request server")
        self.__stop_awaiting_request.set()

        # Send a request to the server to unblock it.
        with Req0(dial=self.__address, send_timeout=2000) as requestor:
            # "x" is a dummy message. Does not work with empty strings or dictionary.
            requestor.send(dumps({"x": "x"}).encode("utf-8"))

        self.exited = True
        self.__request_handler.close()
        self.__worker_thread.join()

    def __send_message(self, message: dict) -> None:
        """Sends a message over pynng.

        Args:
            message (dict): The message to send.
        """
        self.__request_handler.send(dumps(message).encode("utf-8"))

    def __send_err(self, error_code: int, error_message: str) -> None:
        """Sends an error to the client.

        Args:
            error_code (int): The error code.
            error_message (str): The error message.
        """
        response = {
            "status": error_code,
            "payload": error_message,
        }
        self.__send_message(response)

    def __get_next_message(self) -> dict:
        """Gets the next request from the request handler.

        Raises:
            err: If the request handler has closed.

        Returns:
            dict: The request that was received.
        """
        try:
            request_raw = self.__request_handler.recv()
        except Closed as err:
            self.__stop_awaiting_request.set()
            print(err)
        return loads(request_raw)

    def __validate_request_message(self, data: dict):
        """Validates the data received from the client.

        Args:
            data (dict): The data to validate.

        Raises:
            ValidationError: If the data is invalid.
        """
        try:
            validate(data, self.schemas["request"])
        except ValidationError as err:
            self.__send_err(-1, err.message)
            print(err)
            self.stop_server()

    def __validate_response_message(self, message: dict) -> None:
        """Validates a response message.

        Args:
            message (dict): The message to validate.
        """
        try:
            validate(message, self.schemas["response"])
        except ValidationError as err:
            self.__send_err(-1, err.message)
            print(err)
            self.stop_server()

    def __get_config(self) -> None:
        """Sends the config of the vehicle tracker to the client."""
        config = self.__tracker.get_config()

        response = {
            "status": 0,
            "payload": config,
        }
        self.__send_message(response)

        msg = self.__get_next_message()
        self.__validate_response_message(msg)

        if msg["status"] != 0:
            print("Error occurred. Stopping server.")
            print("Error message:", msg["payload"])
            self.stop_server()

    def __set_config(self, message: dict):
        """Sets the config of the vehicle tracker.

        Args:
            message (dict): The message containing the config.
        """
        self.__tracker.set_config(message["payload"])
        self.__send_message({"status": 0})

    def __worker(self) -> None:
        """The worker that handles requests."""
        while not self.__stop_awaiting_request.is_set():
            message = self.__get_next_message()
            if self.__stop_awaiting_request.is_set():
                break
            self.__validate_request_message(message)

            match message["request_type"]:
                case "get_config":
                    self.__get_config()
                case "set_config":
                    self.__set_config(message)
                case _:
                    self.__send_err(-1, "Invalid request type")
                    print("Invalid request type")
                    self.__stop_awaiting_request.set()
        print("Request server stopped")


class PublishingHandler:
    """Exposes an API for the networking of the vehicle tracker.

    Args:
        vehicle_tracking: The vehicle tracker object.
        config_path (Path): The path to the config file.
    """

    __FILE_DIR_PATH = Path(__file__).parent

    def __init__(self, vehicle_tracking, config_path: Path = Path("./vehicle_tracking_config.json")) -> None:
        self.__tracker = vehicle_tracking
        self.__schemas: dict = {}

        to_open = ["request", "response", "tracker_config"]
        for schema in to_open:
            with open(self.__FILE_DIR_PATH / f"schemas/{schema}.json", "r", encoding="utf-8") as file:
                self.__schemas[schema] = load(file)

        with open(config_path, "r", encoding="utf-8") as file:
            config = load(file)
            pynng_conf = config["pynng"]
            self.__pynng_pubs = pynng_conf["publishers"]

        self.stop_awaiting_request = Event()

        self.__position_sender = _PositionSender(
            self.__tracker,
            self.__pynng_pubs["position_sender"]["address"],
            self.__pynng_pubs["position_sender"]["topics"],
        )
        self.__processed_frame_sender = _ProcessedFrameSender(
            self.__tracker, self.__pynng_pubs["processed_image_sender"]["address"]
        )
        self.request_server = _RequestServer(
            self.__tracker, self.__pynng_pubs["request_config_sender"]["address"], self.stop_awaiting_request
        )

    def send_position(self) -> None:
        """Sends the image and real world coordinates of the tracked box over pynng."""
        self.__position_sender.send_position()

    def send_processed_image(self) -> None:
        """Sends the processed frame over pynng."""
        self.__processed_frame_sender.send_processed_frame()
