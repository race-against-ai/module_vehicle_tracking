"""A script to download resources from a webdav server."""
# Copyright (C) 2023, NG:ITL
from pathlib import Path
from typing import Any
from json import load

from webdav4.client import Client


# Constants
CURRENT_DIR = Path(__file__).parent


def main() -> None:
    """Download resources from a webdav server."""
    config_file_path = CURRENT_DIR.parent / "vehicle_tracking_config.json"
    with open(config_file_path, "r", encoding="utf-8") as config_file:
        config: dict[str, Any] = load(config_file)["resource_downloader"]

    webdav_url = config["url"]
    client_name = config["client_name"]
    client_password = config["client_password"]
    client = Client(webdav_url, auth=(client_name, client_password))

    ls_results = client.ls("/large_files", False)
    for file in ls_results:
        if not isinstance(file, str):
            continue
        file_name = file[::-1]
        file_name = file_name[: file_name.find("/")][::-1]
        dir_for_file = CURRENT_DIR / file_name

        print("Downloading: " + file_name)
        client.download_file(file, dir_for_file)
        if Path.exists(dir_for_file):
            print("File downloaded successfully")
        else:
            print("File could not be downloaded")
        print("\n")


if __name__ == "__main__":
    main()
