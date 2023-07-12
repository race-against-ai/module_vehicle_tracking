from webdav4.client import Client

WEBDAV_URL = "https://cloud.ngitl.dev/remote.php/dav/files/raai_download/"

client = Client(WEBDAV_URL, auth=("raai_download", "mJpyehF5M5ehWJQKBsxOcW3ctn2tm4Ip"))

results = client.ls("/")

for result in results:
    print(result["name"])


client.download_file("/Nextcloud.png", "foo.png")


