from setuptools import find_packages, setup
from oak_pc._version import __version__

setup(
    name="oak_pc",
    packages=find_packages(include=["oak_pc"]),
    version=__version__,  # pylint: disable=undefined-variable
    description="A simple point cloud reconstruction tool for Oak-D camera",
    author="Amir Rastar",
    license="Free",
    install_requires=[
        "numpy==1.23.0",
        "depthai==2.16.0.0",
        "open3d==0.15.1",
        "opencv-python==4.6.0.66",
    ],
    # <-- ADD DEPENDENCIES HERE (comma separated)
)
