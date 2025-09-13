import os
from i308_utils import download, unzip


def download_stereo_data():
    download(
        "https://github.com/udesa-vision/i308-resources/raw/refs/heads/main/tutoriales/tutorial_06/stereo_data.zip"
    )
    unzip(
        "stereo_data.zip",
    )

    os.remove("stereo_data.zip")
