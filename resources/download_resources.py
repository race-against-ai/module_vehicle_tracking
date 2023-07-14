# Imports
from typing import List
from webdav4.client import Client
from pathlib import Path


# Constants
WEBDAV_URL = "https://cloud.ngitl.dev/remote.php/dav/files/raai_download/"
CURRENT_DIR = Path(__file__).parent


def main() -> None:
    client = Client(WEBDAV_URL, auth=("raai_download", "mJpyehF5M5ehWJQKBsxOcW3ctn2tm4Ip"))
    ls_results: List[str] = client.ls("/large_files", False)
    for file in ls_results:
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
