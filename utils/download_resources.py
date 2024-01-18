#!/usr/bin/env python
"""Downloads the resources from the webdav server."""
# Copyright (C) 2023, NG:ITL

from pathlib import Path
from typing import Any
from json import load

from webdav4.client import Client


CURRENT_DIR = Path(__file__).parent


def main() -> None:
    """Downloads the resources from the webdav server."""
    with open(CURRENT_DIR.parent / "vehicle_tracking_config.json", "r", encoding="utf-8") as f:
        config: dict[str, Any] = load(f)["resource_downloader"]

    webdav_url = config["url"]
    client_name = config["client_name"]
    client_password = config["client_password"]
    client = Client(webdav_url, auth=(client_name, client_password))

    ls_results = client.ls("/large_files", False)
    if not isinstance(ls_results, list):
        raise TypeError("The ls_results variable should be a list.")

    for file in ls_results:
        file_name = file[::-1]
        file_name = file_name[: file_name.find("/")][::-1]
        dir_for_file = CURRENT_DIR.parent / "resources" / file_name

        print("Downloading: " + file_name)
        client.download_file(file, dir_for_file)
        if Path.exists(dir_for_file):
            print("File downloaded successfully")
        else:
            print("File could not be downloaded")
        print("\n")


if __name__ == "__main__":
    main()
