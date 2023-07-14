# Copyright (C) 2023, NG:ITL
import versioneer
from pathlib import Path
from setuptools import find_packages, setup


def read(fname):
    return open(Path(__file__).parent / fname).read()


setup(
    name="raai_module_vehicle_tracking",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="NGITL",
    author_email="666arnehilbig@gmail.com",
    description=("RAAI Module Vehicle Tracking for tracking the current position of the vehicle."),
    license="GPL 3.0",
    keywords="vehicle tracking",
    url="https://github.com/vw-wob-it-edu-ngitl/raai_module_vehicle_tracking",
    packages=find_packages(),
    long_description=read("README.md"),
    install_requires=["pynng~=0.7.2", "opencv-python~=4.7.0.72", "numpy~=1.24.2", "webdav4~=0.9.8"],
)
