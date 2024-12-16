# Copyright (C) 2023, NG:ITL
from webdav4.client import Client
from typing import Any
from pathlib import Path
from json import load


def main() -> None:
    current_dir = Path(__file__).parent
    with open(current_dir.parent / "vehicle_tracking_config.json", "r") as f:
        config: dict[str, Any] = load(f)["resource_downloader"]

    webdav_url = config["url"]
    client_name = config["client_name"]
    client_password = config["client_password"]
    client = Client(webdav_url, auth=(client_name, client_password))

    ls_results: list[str] = client.ls("/large_files", False)
    for file in ls_results:
        file_name = file[::-1]
        file_name = file_name[: file_name.find("/")][::-1]
        dir_for_file = current_dir / file_name

        print("Downloading: " + file_name)
        client.download_file(file, dir_for_file)
        if Path.exists(dir_for_file):
            print("File downloaded successfully")
        else:
            print("File could not be downloaded")
        print("\n")


if __name__ == "__main__":
    main()
